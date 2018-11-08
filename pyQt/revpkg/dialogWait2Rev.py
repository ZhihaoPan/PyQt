"""
该文件用于处理从平台发来的数据包
1.
head	file	func_ycsyjc	func_yzfl	func_swfl	chsum
cmd(str)	音频数据地址(nfs) (str)	0/1（bool）	0/1（bool）	0/1（bool）	(str，检验数值)
4.
head	file	stop	chsum
control(str)	音频数据地址(nfs) (str)	0/1(bool)	(校验和)
8.
head	file	chsum
rproc_done (str)	音频数据地址(nfs) (str)	(校验和)
"""
import sys, time, os, json
import zmq
from PyQt5.QtWidgets import QApplication, QDialog,QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import pathlib

from revpkg.wait2Rev import Ui_Dialog
from utils.otherUtils import *

# 设置平台IP地址的默认值
# IP4platform = "10.141.221.202"
IP4platform = "192.168.89.159"


class dialogWait2Rev(QDialog, Ui_Dialog):
    def __init__(self, selfCheckSta=200, parent=None):
        super(dialogWait2Rev, self).__init__()
        self.setupUi(self)
        #self.IP4platform = IP4platform
        self.errorMsg = ""
        self.selfCheckSta=selfCheckSta#todo 传值到线程中
        self.plainTextEdit.appendPlainText("上一步自检请求上报信息："+str(self.selfCheckSta))
        # Timer进行计时的反馈
        self.timer1 = QTimer()
        self.work4Zmq= WorkThread4zmq(self.selfCheckSta)
        self.timer1.timeout.connect(self.showCurrentTime)
        #self.timer1.timeout.connect(self.work4Zmq.start)
        # self.timer.setSingleShot(True)#设置为timeout只执行一次
        self.timer1.start(1000)
        self.work4Zmq.trigger.connect(self.showMsg)

        # Timer设置5s进行一次线程的查看网络情况的反馈
        self.timer2 = QTimer()
        self.work4Net = WorkThread4Network()
        self.timer2.timeout.connect(self.work4NetStart)
        self.work4Net.trigger.connect(self.showNetwork)

        # 点击按钮平台IP地址的设定
        self.pushButton_3.clicked.connect(self.ipConfirm)
        self.pushButton_4.clicked.connect(self.ipDefault)

    def showCurrentTime(self):
        currentTime = time.asctime(time.localtime(time.time()))
        self.lineEdit.setText(currentTime)

    # 网络，在界面上显示网络连接是否通畅
    def showNetwork(self, result):
        if result == 0:  # 网络可以ping的通
            self.lineEdit_7.setText("该IP地址的网络通畅(每5s一次测试)")
        else:
            self.lineEdit_7.setText("该IP地址的网络无法Ping通(每5s一次测试)")

    def showMsg(self,Msg):
        if Msg:
            self.lineEdit_2.setText(Msg["file"])
            self.lineEdit_3.setText(str(Msg["func_ycsyjc"]))
            self.lineEdit_4.setText(str(Msg["func_swfl"]))
            self.lineEdit_5.setText(str(Msg["func_yzfl"]))
            self.lineEdit_6.setText(Msg["chsum"])
            self.plainTextEdit.appendPlainText(Msg["ifFileOpen"])
        else:
            self.plainTextEdit.appendPlainText("\n收到的数据包头不是cmd")
    # IP,如果用户点击的是确认
    def ipConfirm(self):
        if self.lineEdit_8.text()=="":
            box = QMessageBox.critical(self, "Wrong", "请输入IP地址", QMessageBox.Ok | QMessageBox.Cancel)
        self.IP4platform = self.lineEdit_8.text()
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + self.IP4platform)
        self.timer2.start()
        self.work4Zmq.start()

    # IP,如果用户点击的是default
    def ipDefault(self):
        self.IP4platform = IP4platform
        self.lineEdit_8.setText(self.IP4platform)
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + IP4platform)
        self.timer2.start()
        self.work4Zmq.start()

    def work4NetStart(self):
        """
        重新设置了timer使得第一次ping的时候是立刻的，后面的ping是每隔10s一次
        :return:
        """
        self.work4Net.start()
        self.timer2.stop()
        self.timer2.start(10000)


    # 如果在自检的时候出现的问题，可以传递错误信息，汇总后一起发送给平台
    # todo 查看需要发送什么样的自检信息到这里
    def setSelfcheckError(self, str):
        pass


class WorkThread4Network(QThread):
    """
    线程：用于Ping网络
    """
    # 暂时返回的是bool的数值
    trigger = pyqtSignal(bool)

    def __init__(self):
        super(WorkThread4Network, self).__init__()

    def setIP(self, strIP):
        self.strIP = strIP

    def run(self):
        strPing = "ping " + self.strIP
        result = os.system(strPing)
        # 返回result
        self.trigger.emit(result)


class WorkThread4zmq(QThread):
    """
    线程,通过zmq和平台进行数据的交互
    """
    trigger = pyqtSignal(dict)

    def __init__(self,selfCheckSta):
        """
        :param selfCheckSta:自检后发来的自检结果
        此时本机作为rep端等待平台发送数据到本机
        """
        super(WorkThread4zmq, self).__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        self.selfCheckSta=selfCheckSta

    def run(self):
        """
        :return:
        """
        self.message = dict(self.socket.recv_json())
        print("Received request: {}".format(self.message))
        # todo 这里要把收到的json进行拆包，首先判断nfs是否可读，判断校验数值是否正确，然后写校验结果和错误信息
        # if head== cmd就回传msg else 错误信息加同时跳出thread
        returnMsg={}
        ifReady=1#用于判断系统是否准备完毕
        error=self.selfCheckSta#记录发生错误的错误码，初试设置默认值为自检的结果码
        if self.message["head"] == "cmd":
            ifReady=ifReady and 1
            #todo 在这边要做文件是否存在能否打开的测试，计算校验数值是否正确
            self.nfs = pathlib.Path(self.message["file"])
            self.func_ycsyjc=self.message["func_ycsyjc"]
            self.func_swfl=self.message["func_swfl"]
            self.func_yzfl=self.message["func_yzfl"]
            self.chsum=self.message["chsum"]

            #首先进行校验值的计算判断发送的内容是否正确
            chstr=getChstr(self.message)
            # 把message中的信息加入回传的字典中
            returnMsg.update(self.message)
            #计算校验码
            chsum=crc32asii(str(chstr))
            if chsum==self.chsum:
                ifReady=ifReady and 1
                #设置信息中校验正确
                returnMsg.update({"chsum":"经过校验后，发送信息内容无误。"})
            else:
                error=600
                ifReady = ifReady and 0
                returnMsg.update({"chsum": "经过校验后，发送信息内容出现错误。"})
            if self.nfs.exists():
                ifReady = ifReady and 1
                #todo 再信息栏中显示该文件夹可以打开,(统计文件夹下的文件数量 放到后面做还是现在做）
                returnMsg.update({"ifFileOpen":str(self.nfs.absolute())+": 该文件目录存在."})
            else:
                ifReady = ifReady and 0
                error=404
                returnMsg.update({"ifFileOpen": str(self.nfs.absolute()) + ": 该文件目录不存在."})
            self.trigger.emit(returnMsg) #通过trigger返回线程信息
        else:
            ifReady = ifReady and 0
            error=700
            returnMsg.update({"Error":"Head is not cmd"})

        # todo 这里要对需要发送的数据进行组包
        sendDic={"head":"rec",
                 "file":str(self.nfs.absolute()),
                 "network":100,
                 "ready":ifReady,
                 "error":error
                 }
        sendChsum=crc32asii(str(sendDic))
        sendDic.update({"chsum":sendChsum})
        self.socket.send_json(sendDic)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = dialogWait2Rev()
    dialog.show()
    sys.exit(app.exec_())
