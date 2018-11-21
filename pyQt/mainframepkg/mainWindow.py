"""
主界面的显示，完成包的持续发送
"""
import zmq,sys,time
from PyQt5.QtWidgets import QMainWindow,QApplication,QMessageBox
from PyQt5.QtCore import QTimer,QThread,pyqtSignal

from mainframepkg.mainFrame import Ui_MainWindow
from utils.getState import *

class windowMainProc(QMainWindow,Ui_MainWindow):
    def __init__(self,fileName,fileNum,parent=None):
        """
        todo
        给监控模块开一个线程一个timer参照之前的得到各个情况，线程监控开一个线程
        （每个线程一个音频？还是每个音频分片给多个线程，倾向于后者，我们先读取一个音频进行静音分片，在对每一片进行处理）
        一个QTimer自动调用音频源的多路采集，多路采集先开三个线程，同时读取音频数据做后续的处理得到每个线程最后的结果
        三个线程中都要有timer用于每5回传主线程一个信息，组包发给平台（组包发包的过程也要在线程中进行）；等三个线程都处理完之后

        :param fileName:
        :param fileNum:
        :param parent:
        """
        super(windowMainProc,self).__init__(parent)
        self.setupUi(self)
        self.lineEdit.setText(fileName)
        self.lineEdit_2.setText(fileNum)
        #设置第一个Timer用于界面显示后的监控模型显示,网络情况每10s一次检测，其他信息每秒检测一次
        self.timer1=QTimer()
        #初始化监控的线程
        self.work4monitor=WorkThread4Monitor()
        self.work4monitor.trigger.connect(self.showMonitor)
        self.timer1.start()
        self.timer1.timeout.connect(self.showCurrentTime)

    def showCurrentTime(self):
        currentTime = time.asctime(time.localtime(time.time()))
        self.lineEdit_11.setText(currentTime)
        self.work4monitor.start()

    def showMonitor(self,monitorMsg):
        self.lineEdit_6.setText(monitorMsg["cpu"])
        self.lineEdit_7.setText(monitorMsg["gpu"])
        self.lineEdit_8.setText(monitorMsg["mem"])
        self.lineEdit_10.setText((monitorMsg["tem"]))
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


if __name__=="__main__":
    app = QApplication(sys.argv)
    Qselfcheck = windowMainProc("1","2")
    Qselfcheck.show()
    sys.exit(app.exec_())