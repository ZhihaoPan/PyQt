import sys
from PyQt5.QtWidgets import QApplication,QDialog,QLineEdit,QMessageBox
from selfcheck.dialogSelfCheck import dialogSelfCheck
from loginpkg.login import Ui_Dialog

class dialogLogin(QDialog,Ui_Dialog):
    def __init__(self,parent=None):
        super(dialogLogin,self).__init__()
        self.setupUi(self)
        self.lineEdit_2.setEchoMode(QLineEdit.Password)
        #创建事件和函数之间的连接
        self.pushButton.clicked.connect(self.logincheck)
        self.pushButton_2.clicked.connect(self.cancel)

    def logincheck(self):
        if self.lineEdit.text().strip()=="admin" and self.lineEdit_2.text().strip()=="admin":
            self.nextwindow = dialogSelfCheck()
            self.nextwindow.show()
            self.close()
        else:
            box=QMessageBox.critical(self,
                            "Wrong",
                            "用户或者密码错误",
                            QMessageBox.Ok|QMessageBox.Cancel)
            self.lineEdit.setText("")
            self.lineEdit_2.setText("")

    def cancel(self):
        self.close()

if __name__=="__main__":
    app=QApplication(sys.argv)
    Qlogin=dialogLogin()
    Qlogin.show()
    sys.exit(app.exec_())