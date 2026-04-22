from PySide6.QtGui import QImage
from PIL import Image

__all__ = [
    'pil2qimage'
]


def pil2qimage(pil_image: Image):
    if pil_image.mode == 'L':
        q_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_Grayscale8)
    elif pil_image.mode == 'RGBA':
        q_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    elif pil_image.mode == 'RGB':
        q_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)
    else:
        return None

    return q_image

