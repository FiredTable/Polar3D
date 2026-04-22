from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent, QPainter
from PySide6.QtCore import Qt, QPoint

__all__ = ['ImageViewer']


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))  # 创建场景
        self._pixmap_item = QGraphicsPixmapItem()  # 创建一个空的pixmap item
        self.scene().addItem(self._pixmap_item)  # 将pixmap item添加到场景中

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏横向滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏纵向滚动条
        self.setDragMode(QGraphicsView.NoDrag)  # 启用拖动

        self._current_width = 0
        self._current_height = 0

    def reset_viewer(self):
        self.scene().clear()  # 清除场景中的内容
        self._pixmap_item = self.scene().addPixmap(QPixmap())  # 添加一个空的pixmap item
        self.resetTransform()
        self._current_width = 0
        self._current_height = 0
        self.setSceneRect(self.scene().itemsBoundingRect())

    def set_image(self, q_image: QImage):
        # 检查宽度和高度是否发生变化
        auto_fit = False
        if q_image.width() != self._current_width or q_image.height() != self._current_height:
            self._current_width = q_image.width()
            self._current_height = q_image.height()
            auto_fit = True  # 如果尺寸变化，则启用auto_fit

        pixmap = QPixmap.fromImage(q_image)  # 将QImage转换为QPixmap
        self._pixmap_item.setPixmap(pixmap)  # 更新pixmap item的图像

        if auto_fit:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)  # 保持宽高比
            self.setSceneRect(self._pixmap_item.boundingRect())  # 根据图像大小调整场景矩形

    def wheelEvent(self, event: QWheelEvent):
        if len(self.scene().items()) == 0:
            return
        cur_point = event.position()
        scene_pos = self.mapToScene(QPoint(cur_point.x(), cur_point.y()))
        view_width = self.viewport().width()
        view_height = self.viewport().height()
        h_scale = cur_point.x() / view_width
        v_scale = cur_point.y() / view_height
        wheel_delta_value = event.angleDelta().y()
        scale_factor = self.transform().m11()
        if (scale_factor < 0.05 and wheel_delta_value < 0) or (scale_factor > 50 and wheel_delta_value > 0):
            return
        if wheel_delta_value > 0:
            self.scale(1.2, 1.2)
        else:
            self.scale(1.0 / 1.2, 1.0 / 1.2)
        view_point = self.transform().map(scene_pos)
        self.horizontalScrollBar().setValue(int(view_point.x() - view_width * h_scale))
        self.verticalScrollBar().setValue(int(view_point.y() - view_height * v_scale))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)
