# from __future__ import print_function
import torch
import torch.optim as optim
from scipy.io import wavfile
import torch.backends.cudnn as cudnn
from torchvision import transforms

from custom_wav_loader import wavLoader,ToTensor
import numpy as np
from models.model import LeNet, VGG, resnet18,resnet50,resnet101
from train import train, test,demo_test
import os
from models.crnn import CRNN, CRNN_GRU
import demo

# import visdom


def run(demo_path):
    allLabels = {'baby_cry': 0, 'car_engine': 1, 'crowd': 2, 'dog_bark': 3, 'gun_shot': 4,
                 'scream': 5, 'siren': 6, 'speaking': 7, 'street_music': 8}
    # vis = visdom.Visdom(use_incoming_socket=False)
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        idx = torch.cuda.current_device()
        print("Current GPU:" + str(idx))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_path = r"/home/panzh/DataSet/Urbandataset/train"
    valid_path = r"/home/panzh/DataSet/Urbandataset/valid"
    test_path = r"/home/panzh/DataSet/Urbandataset/test"
    #demo_path=r"/home/panzh/Downloads/demoAudio/test2.wav"
    demotest=True
    # parameter
    optimizer = 'adadelta'  # adadelta adam SGD
    lr = 0.007  # to do : adaptive lr 0.001
    epochs =300
    epoch = 1
    momentum = 0.9  # for SGD

    iteration = 0
    patience = int(0.25*epochs)
    log_interval = 50

    seed = 1234  # random seed
    batch_size = 1# 100
    test_batch_size = 1
    arc = 'ResNet101'  # LeNet, VGG11, VGG13, VGG16, VGG19' ResNet CRNN
    loaderType="logmeldelta" #logmeldelta, logmel, stft
    # sound setting
    window_size = 0.02  # 0.02
    window_stride = 0.01  # 0.01
    window_type = 'hamming'
    normalize = True

    weight = torch.tensor((0.5, 0.25, 0.25, 0.25, 1, 0.6, 1, 1, 0.16, 0.4)).cuda()
    # loading data
    if demotest:
        #todo 把loader_type作为一个list存不同的处理方式
        demo_dataset=demo.wavLoader(test_path,demo_path,allLabels=allLabels, window_size=window_size, window_stride=window_stride, window_type=window_type,
                                 normalize=normalize,loader_type=loaderType)
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=test_batch_size, shuffle=None, num_workers=1,
                                                  pin_memory=True, sampler=None)
    else:
        train_dataset = wavLoader(train_path, window_size=window_size, window_stride=window_stride, window_type=window_type,
                                  normalize=normalize,loader_type=loaderType)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   pin_memory=True, sampler=None)
        valid_dataset = wavLoader(valid_path, window_size=window_size, window_stride=window_stride, window_type=window_type,
                                  normalize=normalize,loader_type=loaderType)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=None, num_workers=2,
                                                   pin_memory=True, sampler=None)
        test_dataset = wavLoader(test_path, window_size=window_size, window_stride=window_stride, window_type=window_type,
                                 normalize=normalize,loader_type=loaderType)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=None, num_workers=2,
                                                  pin_memory=True, sampler=None)

    # build model
    if arc == 'LeNet':
        model = LeNet()
        print("Using LeNet")
    elif arc.startswith('VGG'):
        model = VGG(arc)
        print("Using VGG")
    elif arc.startswith('ResNet50'):
        model = resnet50()
        print("Using ResNet50")
    elif arc.startswith("ResNet18"):
        model=resnet18()
        print("Using resnet18")
    elif arc.startswith("CRNN"):
        #model = CRNN(nc=1, nh=96)
        model=CRNN_GRU()
    elif arc.startswith("ResNet101"):
        model=resnet101()
        print("Using resnet101")
    else:
        model = LeNet()

    if str(device) == "cuda:1" or str(device)=="cuda:0":
        cuda = True
        model = torch.nn.DataParallel(model.cuda(),device_ids=[1])
        print("Using cuda for model...")
    else:
        cuda = False

    # define optimizer
    if optimizer.lower() == 'adam':  # adadelta
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'adadelta':  # adadelta
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    cudnn.benchmark = True
    best_valid_loss = np.inf

    if os.path.isfile('./checkpoint/' + str(arc) + '.pth'):
        state = torch.load('./checkpoint/' + str(arc) + '.pth')
        print('load pre-trained model of ' + str(arc) + '\n')
        #print(state)
        best_valid_loss = state['acc']
        epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
       	set_lasttime_lr(optimizer, state['lr'])
    # visdom
    # loss_graph = vis.line(Y=np.column_stack([10, 10, 10]), X=np.column_stack([0, 0, 0]),
    #                      opts=dict(title='loss', legend=['Train loss', 'Valid loss', 'Test loss'], showlegend=True,
    #                                xlabel='epoch'))

    # trainint with early stopping

    print('\nStart training...')
    if demotest:
        pre_label=demo_test(demo_loader,model, cuda, mode='Test loss', class2index=demo_dataset.getClass2Index())
    else:
        while (epoch < epochs + 1) and (iteration < patience):
            #调节学习率
            lr = adjust_learning_rate(optimizer, epoch)
            #开始训练
            train(train_loader, model, optimizer, epoch, cuda, log_interval,weight)
            print('train Finished!!')
            #计算训练验证测试的损失
            train_loss = test(train_loader, model, cuda, mode='Train loss', class2index=train_dataset.getClass2Index())
            valid_loss = test(valid_loader, model, cuda, mode='Valid loss', class2index=valid_dataset.getClass2Index())
            test_loss = test(test_loader, model, cuda, mode='Test loss', class2index=test_dataset.getClass2Index())

            #如果验证集的损失比之前来的小就保存
            if valid_loss > best_valid_loss:
                iteration += 1
                print('\nLoss was not improved, iteration {0}\n'.format(str(iteration)))
            else:
                print('\nSaving model of ' + str(arc) + '\n')
                iteration = 0
                best_valid_loss = valid_loss
                state = {'net': arc, 'epoch': epoch, 'state_dict': model.state_dict(), 'acc': valid_loss, 'lr': lr}
                if not os.path.isdir('checkpoint'):  # model load should be
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/' + str(arc) + '.pth')
            epoch += 1
            # vis.line(Y=np.column_stack([train_loss, valid_loss, test_loss]), X=np.column_stack([epoch, epoch, epoch]),
            #         win=loss_graph, update='append',
            #        opts=dict(legend=['Train loss', 'Valid loss', 'Test loss'], showlegend=True))
    print('Finished!!')


def adjust_learning_rate(optimizer, epoch):

    if epoch // 150 == 0:
        # lr=lr*(0.1**(epoch//30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.97
            print("current lr:{:.3f}".format(param_group["lr"]))
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]
    return param_group["lr"]


def set_lasttime_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    import os,time
    path=r"/home/panzh/DataSet/Urbandataset/valid/gun_shot"
    path=r"/home/panzh/Downloads/demoAudio"
    files=os.listdir(path)
    for file in files:
        print("predict filename:{}".format(file))
        run(os.path.join(path,file))

    # run(r"/home/panzh/Downloads/demoAudio/LDC2007S10.wav")


    # from PIL import Image
    # image=Image.open(r"D:\Dataset\Gunshot\x1.png")
    # print(image)#445X257
    # sr,file_data=wavfile.read(r"D:\Dataset\Gunshot\M1.wav")
    # #file_data=np.ones((400))
    # mat=np.zeros((2,200,2))
    # matrix=np.empty_like(mat)
    # for i in range(2):
    #     # print(file_data.shape)
    #     # print(file_data[(batch_size*i):(batch_size+(batch_size*i))].shape)
    #     # print(matrix.shape)
    #     matrix[i] = file_data[(200 * i):(200 + (200 * i))]
    # mat_tensor = torch.from_numpy(matrix)
    # mat_tensor = mat_tensor.float()
    # print(mat_tensor.shape)