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
IP4platform = "localhost"
# IP4platform = "192.168.89.159"


class dialogWait2Rev(QDialog, Ui_Dialog):
    def __init__(self, selfCheckSta=200, parent=None):
        super(dialogWait2Rev, self).__init__()
        self.setupUi(self)
        #self.IP4platform = IP4platform
        self.filepath=""
        self.errorMsg = ""
        self.selfCheckSta=selfCheckSta#todo 传值到线程中
        self.plainTextEdit.appendPlainText("上一步自检请求上报信息："+str(self.selfCheckSta))
        # Timer进行计时的反馈，给ZMQ开了一个线程
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
            self.filepath=Msg["file"]
            # 给发送文件信息开一个线程
            self.work4Send = WorkThread4Send(self.filepath)
            self.work4Send.trigger.connect(self.showFileInfo)
            self.work4Send.start()
        else:
            self.plainTextEdit.appendPlainText("\n收到的数据包头不是cmd")


    def showFileInfo(self,info):
        self.lineEdit_9.setText(str(info["file_num"]))
        self.lineEdit_10.setText(info["time"])
        self.lineEdit_11.setText(info["chsum"])
        self.lineEdit_13.setText(info["head"])
        self.lineEdit_12.setText(info["stop"])

    # IP,如果用户点击的是确认
    def ipConfirm(self):
        if self.lineEdit_8.text()=="":
            box = QMessageBox.critical(self, "Wrong", "请输入IP地址", QMessageBox.Ok | QMessageBox.Cancel)
        self.IP4platform = self.lineEdit_8.text()
        self.plainTextEdit.appendPlainText("设置Ping地址为："+str(self.IP4platform))
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + self.IP4platform)
        self.timer2.start()
        self.work4Zmq.start()
        self.plainTextEdit.appendPlainText("开始网络通信检测....")
        self.plainTextEdit.appendPlainText("开始监听5555端口，等待平台发送数据....")

    # IP,如果用户点击的是default
    def ipDefault(self):
        self.IP4platform = IP4platform
        self.lineEdit_8.setText(self.IP4platform)
        self.plainTextEdit.appendPlainText("设置Ping地址为：" + str(self.IP4platform))
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + IP4platform)
        self.timer2.start()
        self.work4Zmq.start()
        self.plainTextEdit.appendPlainText("开始网络通信检测....")
        self.plainTextEdit.appendPlainText("开始监听5555端口，等待平台发送数据....")

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
        #端口的绑点放在线程中做
        self.selfCheckSta=selfCheckSta

    def run(self):
        """
        :return:
        """
        self.socket.bind("tcp://*:5555")#绑定端口
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

        # todo 这里要对需要发送的数据进行组包，network想个好的办法能够获得上面运行后的数据同时在sendjson之前发送
        sendDic={"head":"rec",
                 "file":str(self.nfs.absolute()),
                 "network":1,
                 "ready":ifReady,
                 "error":error
                 }
        sendChsum=crc32asii(str(sendDic))
        sendDic.update({"chsum":sendChsum})
        #发给平台数据包，直接传输json格式
        self.socket.send_json(sendDic)


class WorkThread4Send(QThread):
    """
    该线程用于计算文件的数量，预处理音频所需要的时长和组成数据包发送给平台
    """
    trigger=pyqtSignal(dict)
    def __init__(self,filepath):
        super(WorkThread4Send,self).__init__()
        self.filepath=filepath
        self.context = zmq.Context()
        self.socket=self.context.socket(zmq.REQ)

    def run(self):
        #组成数据包
        sendMsg={"head":"report",
                 "file":self.filepath}
        file_num=countWavFile(self.filepath)
        sendMsg.update({"file_num":file_num})
        #todo 这里还需要添加计算时间的函数
        sendMsg.update({"time":"00:00：00"})
        chsum=crc32asii(sendMsg)
        sendMsg.update({"chsum":chsum})
        #zmq
        self.socket.connect("tcp://"+IP4platform+":5556")
        print("Sending report....: %s"% str(sendMsg))
        self.socket.send_json(sendMsg)

        self.revMsg=self.socket.recv_json()
        print("Received reply: %s" % (self.revMsg))
        #对收到的Msg进行解析
        #对chsum进行校验
        chsum=self.revMsg["chsum"]
        revChstr=getChstr(self.revMsg)
        revChsum=crc32asii(revChstr)
        #初始化要发送到主线程的信息
        info={"file_num":file_num,
              "time":"00:00:00"}

        if revChsum==chsum:
            #todo 报告收到的数据包校验正确
            info.update({"chsum":"收到的数据包校验正确"})

        if self.revMsg["head"]=="control":
            #返回给界面收到了control信息，可以进行下一步
            info.update({"head":"收到的数据包头为'control'"})
            if self.revMsg["stop"]==0:
                info.update({"stop":"平台要求继续进行后续操作！"})
            else :
                info.update({"stop": "平台要求停止进行后续操作！"})
        self.trigger.emit(info)

def countWavFile(path):
    queue = []
    count = 0
    queue.append(path)
    while len(queue) > 0:
        tmp = queue.pop(0)
        if os.path.isdir(tmp):
            for item in os.listdir(tmp):
                outfile = os.path.join(tmp, item)
                queue.append(outfile)
        elif os.path.isfile(tmp):
            name = os.path.basename(tmp)
            extension = name.split('.')
            if len(extension) == 2:
                if extension[1] == 'wav':
                    count += 1
    return count
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = dialogWait2Rev()
    dialog.show()
    sys.exit(app.exec_())
