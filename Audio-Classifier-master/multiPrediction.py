from __future__ import print_function
import torch
import torch.optim as optim
from scipy.io import wavfile
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from custom_wav_loader import wavLoader,ToTensor
import numpy as np
from models.model import LeNet, VGG, resnet18,resnet50,resnet101
from models.network_resnext import resnext101_32x4d_,resnext101_64x4d_
import os,time
from models.crnn import CRNN, CRNN_GRU
import demo

import torch.nn.functional as F
from torch.autograd import Variable
import torch,time
import numpy as np
# import visdom


def run(demo_path,models):
    # vis = visdom.Visdom(use_incoming_socket=False)
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        idx = torch.cuda.current_device()
        print("Current GPU:" + str(idx))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    test_path = r"/home/panzh/DataSet/Urbandataset/valid"

    demotest=True
    # parameter

    test_batch_size = 1
    loaderType="logmeldelta" #logmeldelta, logmel, stft
    # sound setting
    window_size = 0.02  # 0.02
    window_stride = 0.01  # 0.01
    window_type = 'hamming'
    normalize = True
    allLabels = {'baby_cry': 0, 'car_engine': 1, 'crowd': 2, 'dog_bark': 3, 'gun_shot': 4,
                  'multispeaker': 5, 'scream': 6, 'siren': 7, 'speaking': 8,'stLaughter':9,'telephone'}

    # loading data
    if demotest:
        #todo 把loader_type作为一个list存不同的处理方式
        demo_dataset=demo.wavLoader(test_path,demo_path, allLabels=allLabels,window_size=window_size, window_stride=window_stride, window_type=window_type,
                                 normalize=normalize,loader_type=loaderType)
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=test_batch_size, shuffle=None, num_workers=1,
                                                  pin_memory=True, sampler=None)
    else:
        test_dataset = wavLoader(test_path, window_size=window_size, window_stride=window_stride, window_type=window_type,
                                 normalize=normalize,loader_type=loaderType)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=None, num_workers=4,
                                                  pin_memory=True, sampler=None)
    modellist = {}
    for arc,weight in models.items():
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
            model = resnet18()
            print("Using resnet18")
        elif arc.startswith("CRNN"):
            # model = CRNN(nc=1, nh=96)
            model = CRNN_GRU()
        elif arc.startswith("ResNet101"):
            model = resnet101()
            print("Using resnet101")
        elif arc.startswith("resnext"):
            model = resnext101_32x4d_(pretrained=None)
            print("resnext101_32x4d_")
        else:
            model = LeNet()
        # build model
        if str(device) == "cuda:1" or str(device)=="cuda:0":
            cuda = True
            model = torch.nn.DataParallel(model.cuda(),device_ids=[idx])
            print("Using cuda for model...")
        else:
            cuda = False
        cudnn.benchmark = True
        if os.path.isfile('./checkpoint/' + str(arc) + '.pth'):
            state = torch.load('./checkpoint/' + str(arc) + '.pth')
            print('load pre-trained model of ' + str(arc) + '\n')
            #print(state)
            model.load_state_dict(state['state_dict'])
        modellist.update({model:weight})

    print('\nStart testing...')
    if demotest:
        pre_label=demo_test(demo_loader,modellist, cuda, mode='Test loss', class2index=demo_dataset.getClass2Index())
    else:
        test(test_loader, modellist, cuda, mode='Test loss', class2index=test_dataset.getClass2Index())
        print('Finished!!')

def test(loader, modellist, cuda, mode, class2index, verbose=True):
    for model,weight in modellist.items():
        model.eval()
    xx_loss = 0
    correct = 0
    pre_top2=[]
    wrongdicCount = {}
    FPdicCount={key:0 for key,value in class2index.items()}
    all_labels={key:0 for key,value in class2index.items()}
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data), Variable(target)
        #get target labels
        target_labels=index2Class(class2index ,target)
        for label in target_labels:
            all_labels[label]+=1
        outputs=[]
        for model,weight in modellist.items():
            output=model(data)
            #output=F.softmax(output)
            if len(outputs)==0:
                outputs=output*weight
            else:
                outputs.add_(output*weight)
        outputs = F.softmax(outputs)
        pre_top2.append(accuracy4topK(outputs,target))
        #get prediction labels
        if outputs.data.max(1, keepdim=True)[0]>0.00:
                #print(outputs.data.max(1, keepdim=True)[1])
            pred=outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            print(pred)
        else:
            pred=torch.tensor([[0]]).cuda()

        pred_labels = index2Class(class2index, pred)
        wrongdic, FPdicCount = calWrongAns(target_labels, pred_labels, FPdicCount)
        wrongdicCount = dic_merge(wrongdic, wrongdicCount)
        # FPdicCount=dic_merge(FPdic,FPdicCount)
        # print("target" + str(target_labels))
        # print("output"+str(pred_labels))
        if cuda:
            correct += pred.eq(target.data.view_as(pred)).cuda().sum()
        else:
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #todo 计算重要类别的准确率
    #xx_loss+=1-dic_count["gun_shot"]/dic_count
    if verbose:
        print('{} set: , Accuracy: {}/{} ({:.0f}%)'.format(
              mode, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        #print(pre_top2)
        print("top2 accuracy:{:.2f}%".format(sum(pre_top2)[-1]/len(pre_top2)))
    print("=================Current labels=" + str(len(loader.dataset)) + "==================")
    print_acc1 = "Other Sounds:"
    print_acc2 = "\nImportant Sounds:"
    for key, value in class2index.items():
        if key!="gun_shot" and key!="scream" and key!="speaking":
            print_acc1 += key + ": {}/{} {:.1f}% , FP: {} ;   ".format(all_labels[key]-wrongdicCount[key], all_labels[key],
                                                           (1 - wrongdicCount[key] / all_labels[key]) * 100, FPdicCount[key])
        else:
            print_acc2 += key + ": {}/{} {:.1f}% , FP: {} ;   ".format(all_labels[key] - wrongdicCount[key],
                                                                       all_labels[key],
                                                                       (1 - wrongdicCount[key] / all_labels[key]) * 100,
                                                                       FPdicCount[key])

    print(print_acc1)
    print(print_acc2)

def demo_test(loader, modellist, cuda, mode, class2index, verbose=True):
    start=time.clock()
    for model,weight in modellist.items():
        model.eval()
    dic_count = {}
    for data, begin in loader:
        if cuda:
            data= data.cuda()
        else:
            data =Variable(data)
        outputs = []
        for model, weight in modellist.items():
            output = model(data)
            output = F.softmax(output)
            #print("time{}".format(output))
            if len(outputs) == 0:
                outputs = output * weight
            else:
                outputs.add_(output * weight)
        #print(outputs)
        #output=w1*F.softmax(output1)+w2*F.softmax(output2)#多模型交叉预测
        #get prediction labels
        pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred_labels = index2Class(class2index, pred)
        dic_count.update({"{:.1f}s--{:.1f}s".format(begin[0] / 45037 * 2, begin[0] / 45037 * 2 + 2): pred_labels})
    end=time.clock()
    print("=================Demo Prediction==================time:{:.3f}s".format(end-start))
    print(dic_count)

    #return pred_labels

def index2Class(class2index,index):
    classes = []
    if type(index) is int:
        for key, values in class2index.items():
            if values == index:
                classes.append(key)
    else:
        for indexes in index:
            for key, values in class2index.items():
                if values == indexes:
                    classes.append(key)
    return classes

def calWrongAns(target_labels,pre_labels,FPdicCount):
    labels=np.unique(np.sort(target_labels))
    values=np.zeros(len(labels))
    wrongdic=dict(zip(labels,values))
    #FPdic=dict(zip(labels,values))
    for i in range(len(target_labels)):
        if target_labels[i]!=pre_labels[i]:
            wrongdic[target_labels[i]]+=1
            FPdicCount[pre_labels[i]]+=1
    return wrongdic,FPdicCount

def dic_merge(dic,dic_count):
    for key,values in dic.items():
        if key not in dic_count:
            dic_count[key]=values
        else:
            dic_count[key]+=values
    return dic_count

def accuracy4topK(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #如果多个top将res该为res=[]
        res = 0
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res=(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == "__main__":
    # import os,time
    # path=r"/home/panzh/DataSet/Urbandataset/valid/gun_shot"
    # files=os.listdir(path)
    # for file in files:
    #     print("predict filename:{}".format(file))
    #     run(os.path.join(path,file))

    #run(r"/home/panzh/Downloads/demoAudio/2_3.wav", {"ResNet101": 1})
    run(r"/home/panzh/Downloads/demoAudio/1_73.wav",{"VGG13":0.1,"ResNet101":0.5,"resnext":0.4})


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