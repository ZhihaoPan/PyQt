import sys
import loginpkg
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog


if __name__=="__main__":
    app=QApplication(sys.argv)
    Qdialog_login=QDialog()
    login_ui=loginpkg.Ui_Dialog()

    login_ui.setupUi(Qdialog_login)
    Qdialog_login.show()
    if login_ui.pushButton.click():
        print(login_ui.textEdit.toPlainText())
    sys.exit(app.exec_())