import colorsys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from .BaseDetector import BaseDet
from .yolov7.nets.yolo import YoloBody
from .yolov7.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from .yolov7.utils.utils_bbox import DecodeBox
from .yolov7.utils.utils_rbox import *

from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from .yolov5.utils.plots import Annotator, colors, save_one_box
from .yolov5.utils.torch_utils import select_device, time_sync
from .yolov5.utils.rboxs_utils import poly2rbox, rbox2poly



'''
训练自己的数据集必看注释！
'''
class Detector(BaseDet):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'YoloDeepSort/yolov7/model_data/yolov7_obb_uav.pth',
        "classes_path"      : 'YoloDeepSort/yolov7/model_data/uav_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'YoloDeepSort/yolov7/model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   所使用到的yolov7的版本，本仓库一共提供两个：
        #   l : 对应yolov7
        #   x : 对应yolov7_x
        #------------------------------------------------------#
        "phi"               : 'l',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.6,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.7,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        self.build_config()
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        # self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        # self.net=model
        print('{} model, and classes loaded.'.format(self.model_path))
        # if not onnx:
        #     if self.cuda:
        #         self.net = nn.DataParallel(self.net)
        #         self.net = self.net.cuda()

        

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, model,results,pred_poly,PIL_image,crop = False, count = False):
        #---------------------------------------------------#

        with torch.no_grad():
            top_conf    = results[0][:, 5].cpu()
            # top_rboxes  = results[0][:, :5].cpu()
            top_label   = np.array(results[0][:, 6].cpu(), dtype = 'int32')
            # top_polys   = rbox2poly(top_rboxes)
            top_boxes   = poly2hbb(pred_poly)
            # print('top_boxes',top_boxes)
            # print('top_polys',top_polys)

        #---------------------------------------------------------#
        #   获取数据
        #---------------------------------------------------------#
        pred_boxes = []
        for i, c in list(enumerate(top_label)):
            # predicted_class = self.class_names[int(c)]
            predicted_class ='car'
            poly            = pred_poly[i].cpu().numpy().astype(np.int32)
            box             = top_boxes[i].cpu().numpy().astype(np.int32)
            score           = top_conf[i]
            pred_boxes.append([box, poly, predicted_class, score])

        return PIL_image, pred_boxes


if __name__ == '__main__':
    
    det = Detector()
    