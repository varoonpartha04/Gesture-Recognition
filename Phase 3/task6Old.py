from PyQt5 import QtWidgets
from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys 
from phase3_task2 import task2_main

# to convert .ui file to .py use:  pyuic5 -x task6.ui -o task6UI.py

class MyCustomWidget(QWidget):
    def __init__(self, name, parent=None):
        super(MyCustomWidget, self).__init__(parent)
        self.row = QHBoxLayout()

        self.row.addWidget(QLabel(name))
        self.row.addWidget(QPushButton("yes"))
        self.row.addWidget(QPushButton("no"))

        self.setLayout(self.row)

class Window(QMainWindow): 
  
    def __init__(self): 
        super().__init__() 
        self.setWindowTitle("Python ") 
        self.setGeometry(100, 100, 600, 400) 
        self.UiComponents() 
        self.show() 

    def clicked(self):
        # print(str(self.combo_box.currentText()))
        query_list = self.textEdit.toPlainText().strip().split(",")
        knn_predictions, decision_tree_predictions, ppr_predictions = task2_main(1, query_list)
        self.listWidget.clear()
        if(self.combo_box.currentText() == "knn classifier"): 
            self.label.setText("knn classifier selected")
            for i in knn_predictions:
                self.listWidget.addItem(i)

                # Add to list a new item (item is simply an entry in your list)
                self.item = QListWidgetItem(self.mylist)
                self.mylist.addItem(self.item)
                # Instanciate a custom widget 
                self.row = MyCustomWidget(i)
                self.item.setSizeHint(self.row.minimumSizeHint())
                # Associate the custom widget to the list entry
                self.mylist.setItemWidget(self.item, self.row)


        elif(self.combo_box.currentText() == "Decsision tree classifier"): 
            self.label.setText("decision tree selected")
            for i in decision_tree_predictions:
                self.listWidget.addItem(i)  

        elif(self.combo_box.currentText() == "ppr based classifier"): 
            self.label.setText("ppr classifier selected")
            for i in ppr_predictions:
                self.listWidget.addItem(i)  


        self.update()
    
    def update(self):
        self.label.adjustSize()

    # method for widgets 
    def UiComponents(self): 
        self.label = QtWidgets.QLabel(self)
        self.label.setText("my first label")
        self.label.move(50,50)

        self.combo_box = QComboBox(self) 
        self.combo_box.setGeometry(200, 150, 120, 30) 
        self.geek_list = ["knn classifier", "Decsision tree classifier", "ppr based classifier"] 
        self.combo_box.addItems(self.geek_list) 
        
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setGeometry(QtCore.QRect(190, 60, 331, 87))
        self.textEdit.setObjectName("textEdit")
        
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Click me")
        self.b1.clicked.connect(self.clicked)

        self.e1 = QLineEdit()
        self.e1.setValidator(QIntValidator())
        self.e1.setMaxLength(4)
        self.e1.setAlignment(Qt.AlignRight)
        self.e1.setFont(QFont("Arial",20))

        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setGeometry(QtCore.QRect(500, 310, 256, 192))
        self.listWidget.setObjectName("listWidget")


        # Create the list
        self.mylist = QListWidget(self)

        self.mylist.setGeometry(QtCore.QRect(700, 510, 256, 192))
        


        
        # layout = QVBoxLayout()
        # widgets = [QPushButton, QPushButton]
        # for w in widgets:
        #     layout.addWidget(w())

        # self.widget = QWidget()
        # self.widget.setLayout(layout)

        # # print(text)

# create pyqt5 app 
App = QApplication(sys.argv)
window = Window() 
sys.exit(App.exec()) 