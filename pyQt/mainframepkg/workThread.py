import random
import zmq,sys,time
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow,QApplication,QMessageBox
from PyQt5.QtCore import QTimer,QThread,pyqtSignal,QMutexLocker,QMutex
from utils.getState import *
from utils.otherUtils import *
import logging

class WorkThread4Monitor(QThread):
    trigger = pyqtSignal(dict)
    def __init__(self):
        super(WorkThread4Monitor,self).__init__()


    def run(self):
        cpu=getCPUstate(1)
        gpu=getGPUstate()
        mem=getMemorystate()
        tem=getTemstate()
        dic = dict(zip(["cpu", "gpu", "mem", "tem"], [cpu, gpu, mem, tem]))
        self.trigger.emit(dic)
        # self.quit()
        self.exit()
        return




class WorkThread4SendTempMsg(QThread):
    trigger = pyqtSignal(dict)

    def __init__(self,ip4platform):
        super(WorkThread4SendTempMsg,self).__init__()
        self.ip4platform=ip4platform
        self.lastMsg={}
        self.tmpContent={}
        self.context=zmq.Context(1)
        self.socket=self.context.socket(zmq.PUB)
        #todo 此处的ip地址需要后期看下是否要修改
        self.socket.bind("tcp://" + "127.0.0.1" + ":5557")

    def setSendContent(self, tmpContent):
        self.tmpContent.update(tmpContent)

    def run(self):
        #struct 4 send message
        if not self.tmpContent:
            self.trigger.emit({})
            return
        sendMsg = {"head": "msg"}
        sendMsg.update(self.tmpContent)
        chsum = crc32asii(sendMsg)
        sendMsg.update({"chsum":chsum})
        #如果Message串没有被改变的话就等待3s后在进行一次判断
        if sendMsg==self.lastMsg:
            #todo 此处后面要长时间运行一下，检验是否线程过多程序奔溃
            #time.sleep(5)
            print("Sending Msg dont update....")
            retMsg = {"time": sendMsg["time"], "IP": self.ip4platform, "success": 0, "head": sendMsg["head"],
                      "file": sendMsg["file"], "filedone": sendMsg["filedone"], "time_remain": sendMsg["time_remain"]}
            self.trigger.emit(retMsg)
            return
        time.sleep(1)
        print("Sending processing message......:%s" % str(sendMsg))
        self.socket.send_json(sendMsg)
        self.lastMsg=sendMsg

        #返回主进程信息
        retMsg={"time":sendMsg["time"],"IP":self.ip4platform,"success":1,"head":sendMsg["head"]
                , "file":sendMsg["file"],"filedone":sendMsg["filedone"], "time_remain":sendMsg["time_remain"]}
        self.trigger.emit(retMsg)
        self.quit()

class WorkThread4Audio(QThread):
    trigger=pyqtSignal(dict,int,int)
    def __init__(self,ID,mutex):
        super(WorkThread4Audio, self).__init__()
        self.ThreadID=ID
        self.mutex=mutex

    def run(self):

        try:
            tmp=random.randint(1,100)
            time.sleep(tmp)
            # self.tmpContent = {"file": "/home", "filedone": 0, "time_pass": "000001", "time_remain": "000001",
            #                    "num_ycsyjc": 0, "num_swfl": 0, "num_yzfl": 0,
            #                    "time": time.strftime("%H%M%S", time.localtime())}
            # m_locker = QMutexLocker(self.mutex)
            # print("Current Thread ID is :{}".format(self.ThreadID))
            # time.sleep(5)
            # self.trigger.emit(self.tmpContent, self.ThreadID)

        except Exception as e:
            # m_locker = QMutexLocker(self.mutex)
            # time.sleep(5)
            # self.trigger.emit({"ERROR":e,"ThreadID":self.ThreadID},0)


            self.quit()
            return

        self.rstContent={"url":"/home","file":"filename.wav","file_duration":"{:.2f}s".format(random.randint(1,1000)),
                         "ycsyjc": {
                              "timesteps": [
                                  "00.000s,11.111s",
                                  "11.111s,22.222s",
                                  "22.222s,33.333s"
                              ],
                              "content": [
                                  "boom",
                                  "gun",
                                  "scream"
                              ]
                          },
                         "yzfl": {
                              "timesteps": [
                                  "00.000s,11.111s",
                                  "11.111s,22.222s",
                                  "22.222s,33.333s"
                              ],
                              "content": [
                                  "mandarin",
                                  "english",
                                  "uygur"
                              ]
                          },
                         "swfl": {
                              "timesteps": [
                                  "20.123s,23.456s",
                                  "30.111s,35.222s",
                                  "36.000s,37.000s",
                                  "38.000s,40.000s"
                              ],
                              "content": [
                                  "id00001",
                                  "id00003",
                                  "id00001",
                                  "id00002"
                              ],
                              "newid": [
                                  "id00003"
                              ]
                          }
                        }
        m_locker = QMutexLocker(self.mutex)
        time.sleep(2)
        self.trigger.emit(self.rstContent,self.ThreadID,1)
        self.quit()

class WorkThread4SendResult(QThread):
    trigger = pyqtSignal(dict)

    def __init__(self, ip4platform):
        super(WorkThread4SendResult, self).__init__()
        self.ip4platform = ip4platform
        self.lastMsg = {}
        self.dicContent = {}
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.PUB)
        # todo 此处的ip地址需要后期看下是否要修改
        self.socket.bind("tcp://" + "127.0.0.1" + ":5558")

    def setSendContent(self, dicContent):
        self.dicContent.update(dicContent)

    def run(self):
        # struct 4 send message
        if not self.dicContent:
            self.trigger.emit({})
            return
        sendMsg = {"head": "data"}
        sendMsg.update(self.dicContent)
        # chsum = crc32asii(sendMsg)
        # sendMsg.update({"chsum": chsum})
        # 如果Message串没有被改变的话就等待3s后在进行一次判断
        if sendMsg == self.lastMsg:
            # todo 此处后面要长时间运行一下，检验是否线程过多程序奔溃
            # time.sleep(5)
            print("Sending Msg dont update....")
            retMsg = {}
            self.trigger.emit(retMsg)
            return
        time.sleep(1)
        print("Sending result message......:%s" % str(sendMsg))
        self.socket.send_json(sendMsg)
        self.lastMsg = sendMsg

        # 返回主进程信息
        retMsg = {"time":getCurrentTime(),"IP":self.ip4platform,"success":1,"head":sendMsg["head"],
                  "url":sendMsg["url"],"filename":sendMsg["file"],"ycsyjc":sendMsg["ycsyjc"],"yzfl":sendMsg["yzfl"],"swfl":sendMsg["swfl"]}
        self.trigger.emit(retMsg)
        self.quit()

if __name__ == '__main__':
    pass