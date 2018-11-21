import sys, time
from PyQt5.QtWidgets import QApplication,QDialog
from PyQt5.QtCore import QTimer,QThread,pyqtSignal
from selfcheck import selfCheck
from utils.getState import *

from revpkg.dialogWait2Rev import dialogWait2Rev

class dialogSelfCheck(QDialog, selfCheck.Ui_Dialog):
    def __init__(self,paerent=None):
        super(dialogSelfCheck,self).__init__()
        self.setupUi(self)
        self.selfCheckSta = 200#default

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.start)
        self.timer.setSingleShot(True)
        self.timer.start()

        #实例化一个线程
        self.work=WorkThread()
        #线程全部执行完成之后会调用setText函数
        self.work.trigger.connect(self.setText)

        #按钮事件
        self.pushButton.clicked.connect(self.btnContinue)
        self.pushButton_2.clicked.connect(self.btnReport)

    def start(self):
        #time.sleep(1)
        self.work.start()

    def setText(self,dic):

        self.lineEdit_2.setText(dic["cpu"])
        self.lineEdit_3.setText(dic["mem"])
        self.plainTextEdit_2.setPlainText(dic["tem"])
        self.lineEdit_5.setText(dic["time"])
        self.plainTextEdit.setPlainText(dic["gpu"])

        self.label_6.setText("自检完成")
        #QApplication.processEvents()

    def btnContinue(self):
        self.nextwindow=dialogWait2Rev(self.selfCheckSta)
        self.nextwindow.show()
        self.close()

    def btnReport(self):
        #500表示自检出现问题
        self.selfCheckSta=500

#用一个trigger函数用来接收信号，同时可以返回pyqtSignal的参数给上面trigger.connect里的函数
class WorkThread(QThread):
    trigger=pyqtSignal(dict)

    def __init__(self):
        super(WorkThread,self).__init__()

    def run(self):
        #把耗时的放在线程中处理
        #time.sleep(5)
        cutime = getCurrentTime()
        cpu = getCPUstate(1)
        gpu = getGPUstate()
        mem = getMemorystate()
        tem = getTemstate()
        dic=dict(zip(["time","cpu","gpu","mem","tem"],[cutime, cpu, gpu, mem, tem]))
        self.trigger.emit(dic)

if __name__=="__main__":
    app=QApplication(sys.argv)
    Qselfcheck=dialogSelfCheck()
    Qselfcheck.show()
    sys.exit(app.exec_())
