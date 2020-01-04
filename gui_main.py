from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import sys


class MainWindows(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # mainwindows
        self.resize(1200, 600)
        self.setFont(QtGui.QFont("微軟正黑體", 12))
        self.center()
        self.setWindowTitle("Vertebra Segmentation")

        # widget
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        hbox = QtWidgets.QHBoxLayout()
        self.widget.setLayout(hbox)

        v1 = QtWidgets.QVBoxLayout()
        v1.setAlignment(QtCore.Qt.AlignHCenter)
        label_origin = QtWidgets.QLabel("原始圖片", self)
        label_origin.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_origin = QtWidgets.QLabel("Picture", self)
        v1.addWidget(label_origin)
        v1.addWidget(self.picture_origin)
        hbox.addLayout(v1)

        v2 = QtWidgets.QVBoxLayout()
        v2.setAlignment(QtCore.Qt.AlignHCenter)
        label_truth = QtWidgets.QLabel("Ground Truth", self)
        label_truth.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_truth = QtWidgets.QLabel("Picture", self)
        v2.addWidget(label_truth)
        v2.addWidget(self.picture_truth)
        hbox.addLayout(v2)

        v3 = QtWidgets.QVBoxLayout()
        v3.setAlignment(QtCore.Qt.AlignHCenter)
        label_predict = QtWidgets.QLabel("預測圖片", self)
        label_predict.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_predict = QtWidgets.QLabel("Picture", self)
        v3.addWidget(label_predict)
        v3.addWidget(self.picture_predict)
        hbox.addLayout(v3)

        # toolbar
        self.open_image = QtWidgets.QAction('開啟圖片', self)
        self.open_image.triggered.connect(self.load_input)
        self.open_truth = QtWidgets.QAction('開啟Groud Truth', self)
        self.open_truth.triggered.connect(self.load_truth)
        self.predict = QtWidgets.QAction('預測', self)

        self.toolbar = self.addToolBar('Toolbar')
        self.toolbar.addAction(self.open_image)
        self.toolbar.addAction(self.open_truth)
        self.toolbar.addAction(self.predict)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

    def get_file(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", ".", "Image files (*.jpg *.png)")
        return fname

    def load_input(self):
        path = self.get_file()[0]
        if path:
            self.input = rgb2gray(io.imread(path))
            img = QtGui.QImage(self.input, self.input.shape[1], self.input.shape[0], QtGui.QImage.Format_Grayscale8)
            img = img.scaled(250, 600)
            pixmap = QtGui.QPixmap(img)
            self.picture_origin.setPixmap(pixmap)
            self.picture_origin.show()

    def load_truth(self):
        path = self.get_file()[0]
        if path:
            self.truth = rgb2gray(io.imread(path))
            img = QtGui.QImage(self.truth, self.truth.shape[1], self.truth.shape[0], QtGui.QImage.Format_Grayscale8)
            img = img.scaled(250, 600)
            pixmap = QtGui.QPixmap(img)
            self.picture_truth.setPixmap(pixmap)
            self.picture_truth.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    windows = MainWindows()
    sys.exit(app.exec_())
