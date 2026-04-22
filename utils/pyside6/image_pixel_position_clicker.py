import numpy as np
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from ..common import resize_pil_image, np2pil
from .pil2qimage import pil2qimage

__all__ = [
    'ImagePixelPositionClicker'
]


class ImagePixelPositionClicker(QDialog):
    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

        scale_ratio = 1
        while (image.shape[0] / scale_ratio > 1200) or (image.shape[1] / scale_ratio > 800):
            scale_ratio += 1

        self.scale_ratio = scale_ratio
        self.pil_image = resize_pil_image(np2pil(self.image), self.scale_ratio)

        self.setWindowTitle('Click to get pixel position')
        self.setGeometry(200, 200, self.pil_image.width, self.pil_image.height)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(pil2qimage(self.pil_image)))

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.coord_x = None
        self.coord_y = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.coord_x = self.scale_ratio * event.x()
            self.coord_y = self.scale_ratio * event.y()
            # 设置模态对话框的返回值为 QDialog.Accepted，并关闭模态对话框，退出 exec() 的事件循环
            self.accept()
