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
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from revpkg.wait2Rev import Ui_Dialog

# 设置平台IP地址的默认值
IP4platform = "10.141.221.202"


class dialogWait2Rev(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(dialogWait2Rev, self).__init__()
        self.setupUi(self)
        self.IP4platform = IP4platform
        self.errorMsg = ""

        # Timer进行计时的反馈
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.showCurrentTime)
        # self.timer.setSingleShot(True)#设置为timeout只执行一次
        self.timer1.start(1000)

        # Timer设置5s进行一次线程的查看网络情况的反馈
        self.timer2 = QTimer()
        self.work4Net = WorkThread4Network()
        self.timer2.timeout.connect(self.work4Net.start)
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

    # IP,如果用户点击的是确认
    def ipConfirm(self):
        self.IP4platform = self.lineEdit_8.text()
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + self.IP4platform)
        self.timer2.start(5000)

    # IP,如果用户点击的是default
    def ipDefault(self):
        self.IP4platform = IP4platform
        self.lineEdit_8.setText(self.IP4platform)
        # 设置线程中的IP address
        self.work4Net.setIP(self.IP4platform)
        print("set IP4platform IP:" + IP4platform)
        self.timer2.start(5000)

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
    trigger = pyqtSignal(dict)

    def __init__(self):
        super(WorkThread4zmq, self).__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    def run(self):
        self.message = dict(self.socket.recv_json())
        print("Received request: {}".format(self.message))
        # todo 这里要把收到的json进行拆包，首先判断nfs是否可读，判断校验数值是否正确，然后写校验结果和错误信息
        # if head== cmd就回传msg else 错误信息加同时跳出thread谢谢你的爱1111
        if self.message["head"] == "cmd":
            self.trigger.emit(self.message)
        else:
            pass

        # 判断文件是否存在 存在话在 if 校验正确其他也无误 组包 else 错误加 else 错误加 组包  # todo 这里要对需要发送的数据进行组包

        # self.socket.send_json()

    def setSendMsg(self, msg):
        """
        :param msg:发送字典数据
        """
        self.msg = msg


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = dialogWait2Rev()
    dialog.show()
    sys.exit(app.exec_())  # result=os.system(u"ping www.baidu.com")  # print(result)
