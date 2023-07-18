'''
Author: Egrt
Date: 2021-12-06 12:40:16
LastEditors: Egrt
LastEditTime: 2023-02-12 15:37:03
FilePath: \yolox-deepsort\YoloDeepSort\tracker.py
'''
from .strong_sort.strong_sort import StrongSORT
import torch
import numpy as np
from PIL import ImageDraw, ImageFont

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
deepsort = StrongSORT()


def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(image, rbox, font):

    for poly, cls_name, track_id in rbox:
        color = compute_color_for_labels(track_id)
        label = '{}{:d}'.format(cls_name, track_id)
        draw = ImageDraw.Draw(image)
        label = label.encode('utf-8')
        text_origin = np.array([poly[0], poly[1]], np.int32)
        # print('poly----------------------------',poly)
        draw.polygon(xy=list(poly), outline=color, width=2)
        draw.text(text_origin, str(label,'UTF-8'), fill=color, font=font)
        del draw
    return image

def update_tracker(target_detector, image,model,pred,pred_poly,PIL_image):

    new_faces = []
    image_origin, bboxes = target_detector.detect_image(image,model,pred,pred_poly,PIL_image)

    bbox_xywh = []
    polys     = []
    confs     = []
    clss      = []

    for box, poly, cls_id, conf in bboxes:
        bbox_xywh.append(box)
        polys.append(poly)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    outputs = deepsort.update(xywhs, polys, confss, clss, image)
    # print('outputs',outputs)


    rboxes2draw = []
    current_ids = []
    for value in list(outputs):
        box, poly, track_id, cls_, _= value
        # x1, y1, x2, y2 = box
        # x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        rboxes2draw.append((poly, cls_, track_id))
        current_ids.append(track_id)


    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)
    font  = ImageFont.truetype(font='YoloDeepSort/yolov7/model_data/Arial.ttf',
            size=np.floor(3e-2 * image_origin.size[1] + 0.5).astype('int32'))
    image = draw_boxes(image_origin, rboxes2draw, font)




    return image, new_faces, rboxes2draw
