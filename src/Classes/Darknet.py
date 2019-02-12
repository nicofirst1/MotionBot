import logging
from threading import Thread
from Paths import Path as pt
import cv2
from yolo.pydarknet import Detector, Image

logger = logging.getLogger('darknet')


class Darknet(Thread):

    def __init__(self):
        super().__init__()
        self.net = Detector(bytes(pt.join(pt.DARKNET_DIR,"cfg/yolov3.cfg"), encoding="utf-8"),
                            bytes(pt.join(pt.DARKNET_DIR,"weights/yolov3.weights"), encoding="utf-8"), 0,
                            bytes(pt.join(pt.DARKNET_DIR,"cfg/coco.data"), encoding="utf-8"))

    def detect(self, img):
        dark_img = Image(img)
        results = self.net.detect(dark_img)

        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0),
                          thickness=2)
            cv2.putText(img, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

        return results
