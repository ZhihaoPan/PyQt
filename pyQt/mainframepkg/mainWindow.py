"""
主界面的显示，完成包的持续发送
"""
import random

import zmq,sys,time
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow,QApplication,QMessageBox
from PyQt5.QtCore import QTimer,QThread,pyqtSignal,QMutexLocker,QMutex
from mainframepkg.mainFrame import Ui_MainWindow
from mainframepkg.workThread import *
from utils.getState import *



class windowMainProc(QMainWindow,Ui_MainWindow):
    def __init__(self,fileName,fileNum,procFunctions,IP4platform="127.0.0.1",parent=None):
        """
        todo
        给监控模块开一个线程一个timer参照之前的得到各个情况（完成），线程监控开一个线程
        （每个线程一个音频？还是每个音频分片给多个线程，倾向于后者，我们先读取一个音频进行静音分片，在对每一片进行处理）
        一个QTimer自动调用音频源的多路采集，多路采集先开三个线程，同时读取音频数据做后续的处理得到每个线程最后的结果
        三个线程中都要有timer用于每5回传主线程一个信息，组包发给平台（组包发包的过程也要在线程中进行）；等三个线程都处理完之后


        :param fileName: 文件名
        :param fileNum: 文件数量
        :param procFunctions: 准备使用的算法
        :param IP4platform: zmq需要连接的IP地址
        :param parent:
        """
        super(windowMainProc,self).__init__(parent)
        self.setupUi(self)
        self.lineEdit.setText(fileName)
        self.lineEdit_2.setText("{:}".format(fileNum))
        self.textEdit_4.setText("就绪")
        self.textEdit_5.setText("就绪")
        self.textEdit_6.setText("就绪")
        #初始化部分成员变量
        self.procFunctions=procFunctions
        self.ip4platform=IP4platform
        self.tmpContent={}
        self.rstContent={}
        #设置第一个Timer用于界面显示后的监控模型显示,网络情况每10s一次检测，其他信息每秒检测一次
        self.timer1=QTimer()
        #初始化监控的线程
        self.work4monitor=WorkThread4Monitor()
        self.work4monitor.trigger.connect(self.showMonitor)
        self.timer1.start(1000)
        self.timer1.timeout.connect(self.showCurrentTime)

        #该timer用于发送每5s一次的信息
        # self.timer2=QTimer()
        # self.timer2.setSingleShot(1)
        # self.timer2.start(5000)
        # self.timer2.timeout.connect(self.sendTempMsg)
        #该线程只初始化一次
        self.work4SendTempMsg = WorkThread4SendTempMsg(self.ip4platform)
        self.work4SendRstMsg = WorkThread4SendResult(self.ip4platform)
        #该timer用于音频处理块的开始
        self.timer3=QTimer()
        self.timer3.setSingleShot(1)
        self.timer3.start()
        self.timer3.timeout.connect(self.procAudio)

        #给线程的部分内容加锁，保证线程不会混乱
        self.mutex=QMutex()

        #用一个Timer定时清除缓存信息
        self.timer4=QTimer()
        self.timer4.start(100000000)
        self.timer4.timeout.connect(self.clearup)

    def clearup(self):
        self.plainTextEdit.clear()

    #处理音频信息块
    def procAudio(self):
        """
        todo 此处为算法处理模块算法每5s的处理结果存入dicContent中，
        #有个问题如果这三个同时调用了setDicContent会不会混乱,答案是会，当同时调用一个函数的时候就会冲突
        因此需要设置一个mutex进行一个互斥
        设置三个线程三个timer
        :return:
        """
        #这里做一个循环的话
        # while 1:
        #     pass
        self.audioThread1=WorkThread4Audio(ID=1, mutex=self.mutex)
        self.audioThread2=WorkThread4Audio(ID=2, mutex=self.mutex)
        self.audioThread3=WorkThread4Audio(ID=3, mutex=self.mutex)
        self.audioThread1.trigger.connect(self.setContent)
        self.audioThread2.trigger.connect(self.setContent)
        self.audioThread3.trigger.connect(self.setContent)
        self.audioThread1.start(1)
        self.audioThread2.start(2)
        self.audioThread3.start(3)


    def setContent(self, Content, threadID, flag):
        """
        :param Content:
        :param threadID:
        :param flag: 如果flag为0则为5s一次信息，flag为1则为result信息
        :return:
        """
        #(ugly)这里对出来的线程进行判断，执行完了线程就重开
        if threadID is 1:
            #self.textEdit_4.setText("就绪")
            self.audioThread1.start()
            self.textEdit_4.setText("运行")
        elif threadID is 2:
            #self.textEdit_5.setText("就绪")
            self.audioThread2.start()
            self.textEdit_5.setText("运行")
        elif threadID is 3:
            #self.textEdit_6.setText("就绪")
            self.audioThread3.start()
            self.textEdit_6.setText("运行")
        elif threadID is 0:
            self.plainTextEdit.appendPlainText("当前线程：{} 处理音频时发生错误:{}\n".format(Content["ThreadID"], Content["ERROR"]))

        if flag is 0:
            self.plainTextEdit.appendPlainText("当前音频处理线程:{},准备发送5s信息...".format(threadID))
            self.tmpContent.update(Content)
            self.sendTempMsg()
        elif flag is 1:
            self.plainTextEdit.appendPlainText("当前音频处理线程:{},处理完成,准备发送处理结果信息...".format(threadID))
            self.rstContent.update(Content)
            self.sendRstMsg()

    #发送处理信息部
    def sendTempMsg(self):
        """
        发送每5s的信息,初始化的时候要传入内容参数和IP参数
        :return:
        """
        #每5s设置一次content
        self.work4SendTempMsg.setSendContent(self.tmpContent)
        #设置了发送的信息可以开始线程
        self.work4SendTempMsg.start()
        self.work4SendTempMsg.trigger.connect(self.showTmpMsg)

    def sendRstMsg(self):
        """
        发送线程处理的总结果
        :return:
        """
        self.work4SendRstMsg.setSendContent(self.rstContent)
        self.work4SendRstMsg.start()
        self.work4SendRstMsg.trigger.connect(self.showRstMsg)

    #界面显示信息部分
    def showCurrentTime(self):

        currentTime = time.asctime(time.localtime(time.time()))
        self.lineEdit_11.setText(currentTime)
        self.work4monitor.start()



    def showMonitor(self,monitorMsg):
        """
        显示监控信息
        :param monitorMsg:
        :return:
        """
        self.lineEdit_6.setText(monitorMsg["cpu"])
        self.textEdit.setText(monitorMsg["gpu"])
        self.textEdit_3.setText(monitorMsg["mem"])
        self.textEdit_2.setText((monitorMsg["tem"]))

    def showTmpMsg(self, retMsg):
        """
        本地发送数据给平台后在此写入日志同时显示在界面上的框内
        :param retMsg:
        :return:
        """
        if not retMsg:
            self.plainTextEdit.appendPlainText("发送平台的信息包头为:msg,但发送的内容为空!!!!!!!")
            # 写入日志文件
            mainlog("发送平台的信息包头为:msg,但发送的内容为空!!!!!!!",level="warning")
            self.work4SendTempMsg.disconnect()
            return
        text="发送平台信息包头:{}, 当前处理的音频文件是:{}" \
             "\n发送时刻(hhmmss):{}, IP地址:{}" \
             " \n是否发送成功:{}".format(retMsg["head"], retMsg["file"],retMsg["time"], retMsg["IP"], retMsg["success"])
        self.plainTextEdit.appendPlainText(text)
        # 写入日志文件
        #mainlog(text)
        self.lineEdit_3.setText("{}".format(retMsg["filedone"]))
        self.lineEdit_5.setText(retMsg["time_remain"])
        #todo 为什么不加disconnect就会一次性弹出很多
        self.work4SendTempMsg.disconnect()

    def showRstMsg(self,retMsg):
        """
                本地发送数据给平台后在此写入日志同时显示在界面上的框内
                :param retMsg: Return Message
                :return:
                """
        if not retMsg:
            self.plainTextEdit.appendPlainText("发送平台的信息包头为:data,但发送的内容重复或者为空")
            # 写入日志文件
            mainlog("发送平台的信息包头为:data,但发送的内容重复或者为空!!!!!!!", level="warning")
            self.work4SendRstMsg.disconnect()
            return
        text="发送平台信息包头:{}, 当前处理的音频文件是:{} \n发送时刻(hhmmss):{}, IP地址:{}" \
             ", 是否发送成功:{}, 发送内容查询log".format(retMsg["head"], retMsg["filename"], retMsg["time"],
                                                                    retMsg["IP"], retMsg["success"])
        self.plainTextEdit.appendPlainText(text)
        #写入日志文件
        mainlog("{},ycsy:{}\n,swfl:{}\n,yzfl:{}".format(text,retMsg["ycsyjc"],retMsg["swfl"],retMsg["yzfl"]),"debug")
        self.work4SendRstMsg.disconnect()


if __name__=="__main__":
    app = QApplication(sys.argv)
    Qselfcheck = windowMainProc("1","2","127.0.0.1")
    Qselfcheck.show()
    sys.exit(app.exec_())
