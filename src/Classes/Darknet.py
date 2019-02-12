import logging
from threading import Thread

import cv2

from Path import Path as pt
from darknet.python.darknet import load_net, load_meta, detect

logger = logging.getLogger('darknet')


class Darknet(Thread):

    def __init__(self, use_coco):
        super().__init__()

        self.min_score = 0.6
        self.color = (255, 0, 0)

        if use_coco:
            cfg = pt.join(pt.DARKNET_DIR, "cfg/yolov3.cfg")
            weights = pt.join(pt.WEIGHTS_DIR, "yolov3.weights")
            data = pt.join(pt.DARKNET_DIR, "cfg/coco.data")

        else:
            cfg = pt.join(pt.DARKNET_DIR, "cfg/yolov3-openimages.cfg")
            weights = pt.join(pt.WEIGHTS_DIR, "yolov3-openimages.weights")
            data = pt.join(pt.DARKNET_DIR, "cfg/openimages.data")

        cfg = cfg.encode('utf-8')
        weights = weights.encode('utf-8')
        data = data.encode('utf-8')

        self.net = load_net(cfg, weights, 0)
        self.meta = load_meta(data)

    def detect_video(self, img_list):
        """
        Perform detection on list of frames
        :param img_list: list of numpy frames
        :return: a list of lists [i,j]: the element[i][j] is the j-th object detected in the i-th frame
        """
        res = []
        for img in img_list:
            res.append(self.detect_img(img))

        return res

    def detect_img(self, img):
        """
        Detect an image using darknet
        :param img: the image as an np array
        :return: list of detected objects
        """

        try:
            # list of objects with (cat, score, bounds)
            bbs = detect(self.net, self.meta, img, thresh=self.min_score)

        except Exception as e:
            logger.error(e)
            print(e)
            return (img, [])

        return (img, bbs)

    def draw_bounds(self, img, bounds, category, thickness=2):
        """
        Draw bounding boxes onto image
        :param img: the image
        :param bounds: the bounding boxes
        :param category: the cat of the image
        :param color: (tuple of three ints) the color of the bounding boxes
        :param thickness: (int) the thickness of the bb
        :return:
        """
        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), self.color,
                      thickness=thickness)

        cv2.putText(img, str(category.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

    def draw_bounds_list(self, img_list):

        frames = []
        for frame, bb in img_list:
            cat, _, bounds = bb
            self.draw_bounds(frame, bounds, cat)
            frames.append(frame)

        return frames
