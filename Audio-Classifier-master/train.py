from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
from custom_wav_loader import wavLoader
import torch,time
import numpy as np
def train(loader, model, optimizer, epoch, cuda, log_interval, weight,verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if cuda:
            loss = F.cross_entropy(output, target).cuda()
        else:
            loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.item()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(loader.dataset), 100.* batch_idx / len(loader), loss.item()))
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, mode, class2index, verbose=True):
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

        output = model(data)
        #计算显示的时候用softmax；计算交叉熵的时候用原始值
        outputs = F.softmax(output)
        pre_top2.append(accuracy4topK(outputs,target))
        if cuda:
            xx_loss += F.cross_entropy(output, target, size_average=False).cuda().item()   # sum up batch loss
        else:
            xx_loss += F.cross_entropy(output, target, size_average=False).item()
        #get prediction labels
        pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred_labels = index2Class(class2index, pred)
        wrongdic,FPdicCount = calWrongAns(target_labels, pred_labels,FPdicCount)
        wrongdicCount=dic_merge(wrongdic,wrongdicCount)

        if cuda:
            correct += pred.eq(target.data.view_as(pred)).cuda().sum()
        else:
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    #todo 计算重要类别的准确率
    #xx_loss+=1-dic_count["gun_shot"]/dic_count
    xx_loss /= len(loader.dataset)
    if verbose:
        print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
              mode, xx_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))

        print("top2 accuracy:{:.2f}%".format(sum(pre_top2)[-1]/len(pre_top2)))
    print("=================Current labels=" + str(len(loader.dataset)) + "==================")
    print_acc1 = "Other Sounds:"
    print_acc2 = "\nImportant Sounds:"
    for key, value in class2index.items():
        if key!="gun_shot" and key!="scream" and key!="speaking" and key!="multispeaker":
            print_acc1 += key + ": {}/{} {:.1f}% ,: {:.1f} %;   ".format(all_labels[key]-wrongdicCount[key], all_labels[key],
                                                           (1 - wrongdicCount[key] / all_labels[key]) * 100,
                                                           (all_labels[key]-wrongdicCount[key])/(all_labels[key]-wrongdicCount[key]+FPdicCount[key])*100)
        else:
            print_acc2 += key + ": {}/{} {:.1f}% ,: {:.1f} %;   ".format(all_labels[key] - wrongdicCount[key],
                                                                       all_labels[key],
                                                                       (1 - wrongdicCount[key] / all_labels[key]) * 100,
                                                                        (all_labels[key] - wrongdicCount[key]) / (all_labels[key] - wrongdicCount[key] + FPdicCount[key])*100)
            #xx_loss +=FPdicCount[key]/(FPdicCount[key]+all_labels[key] - wrongdicCount[key])

    print(print_acc1)
    print(print_acc2)
    return xx_loss

def demo_test(loader, model, cuda, mode, class2index, verbose=True):
    start=time.clock()
    model.eval()
    dic_count = {}
    for data, begin in loader:
        if cuda:
            data= data.cuda()
        else:
            data =Variable(data)

        output = model(data)
        output=F.softmax(output)
        print(output)
        #get prediction labels
        if output.data.max(1, keepdim=True)[0]>0.90:
                #print(outputs.data.max(1, keepdim=True)[1])
            pred=output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            print(pred)
        else:
            pred=torch.tensor([[0]]).cuda()
        #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred_labels = index2Class(class2index, pred)
        dic_count.update({"{:.1f}s--{:.1f}s".format(begin[0] / 45037 * 2, begin[0] / 45037 * 2 + 2): pred_labels})
    end=time.clock()
    print("=================Demo Prediction==================time:{:.3f}s".format(end-start))
    print(dic_count)

    #return pred_labels

def index2Class(class2index,index):
    classes = []
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

