from PySide6.QtWidgets import QDialog, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

__all__ = [
    'MatplotlibDialog'
]


class MatplotlibDialog(QDialog):
    def __init__(self, window_title: str):
        super().__init__()
        self.setWindowTitle(window_title)

        # 创建 Matplotlib 画布
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def show(self):
        self.figure.tight_layout()
        self.canvas.draw()
        self.exec()
