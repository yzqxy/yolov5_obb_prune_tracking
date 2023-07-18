'''
Author: Egrt
Date: 2021-12-06 12:40:15
LastEditors: [egrt]
LastEditTime: 2023-02-12 13:23:50
FilePath: \yolox-deepsortd:\notebook\yolox-deepsort\YoloDeepSort\BaseDetector.py
'''
from YoloDeepSort.Tracker import update_tracker
import cv2


class BaseDet(object):

    def __init__(self):

        self.stride = 1

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im,model,pred,pred_poly,PIL_image):

        retDict = {
            'frame': None,
            'objects': None,
            'list_of_ids': None,
            'object_bboxes': []
        }
        self.frameCounter += 1
        # im=cv2.imread(im+'/car.jpg')
        im, objects, object_rboxes = update_tracker(self, im,model,pred,pred_poly,PIL_image)

        retDict['frame'] = im
        retDict['objects'] = objects
        retDict['object_rboxes'] = object_rboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
