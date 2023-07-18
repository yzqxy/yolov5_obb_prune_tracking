'''
Description: 
Author: Egrt
Date: 2023-02-12 14:20:19
LastEditors: Egrt
LastEditTime: 2023-02-12 17:42:01
'''
#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import torch
import cv2
import numpy as np
from PIL import Image,ImageDraw
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import math
from YoloDeepSort.AIDetector import Detector
from YoloDeepSort.yolov5.models.common import DetectMultiBackend

from YoloDeepSort.yolov5.models.common import DetectMultiBackend
from YoloDeepSort.yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from YoloDeepSort.yolov5.utils.general import (LOGGER, non_max_suppression_obb,  scale_polys)
from YoloDeepSort.yolov5.utils.plots import Annotator, colors, save_one_box
from YoloDeepSort.yolov5.utils.torch_utils import select_device, time_sync
from YoloDeepSort.yolov5.utils.rboxs_utils import poly2rbox, rbox2poly

def save_image(image, address, num):
    pic_address = address + str(num) + '.jpg'
    cv2.imwrite(pic_address, image)


def coordinate_convert_r(box):  # 将(x,y,w,h,angle)转为四个角点的坐标
    w, h = box[2:-1]
    theta = -box[-1]
    x_lu, y_lu = -w/2, h/2  # 将检测框的中点坐标表示出来
    x_ru, y_ru = w/2, h/2
    x_ld, y_ld = -w/2, -h/2
    x_rd, y_rd = w/2, -h/2

    x_lu_ = math.cos(theta)*x_lu + math.sin(theta)*y_lu + box[0]
    y_lu_ = -math.sin(theta)*x_lu + math.cos(theta)*y_lu + box[1]

    x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
    y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

    x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
    y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

    x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
    y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

    convert_box = [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

    return convert_box

def iou_eight(bbox, candidates):

    a = np.array(bbox).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    b = np.array(candidates).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)

        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


if __name__ == "__main__":
    detector = Detector()
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "test_video/test12.avi"
    video_save_path = "test_video/output.mp4"
    video_fps       = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half='False'
    weights='runs/car/weights/last.pt'    
    model = DetectMultiBackend(weights, device=device, dnn=False)
    model.model.half() if half else model.model.float()


    capture = cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_video     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    j=0

    while(True):
        
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        j=j+1
        # 格式转变，BGRtoRGB
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        save_path='test_video/video_img/'
        save_image(frame, save_path, j)
        print('图片保存地址：', 'video/' + str(j) + '.jpg')
        
        images_name='{}.jpg'.format(j)
        PIL_image=Image.open(save_path+images_name)
        images=cv2.imread(save_path+images_name)
 
        dataset = LoadImages(save_path+'{}'.format(j)+'.jpg', img_size=640)
        
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model(im)
            # NMS
            pred = non_max_suppression_obb(pred, 0.5, 0.2, multi_label=True, max_det=1000)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0s.shape)



        # 进行跟踪检测，输出跟踪帧
        t1 = time.time()
        frame = np.array(detector.feedCap(images,model,pred,pred_poly,PIL_image)['frame'])
        fps  = 1./(time.time()-t1)
        print("fps_track= %.2f"%(fps))
        
        #输出跟踪框
        out = np.array(detector.feedCap(images,model,pred,pred_poly,PIL_image)['object_rboxes'])

        #存储检测框的图
        for poly in pred_poly:
            color = (0,0,0)
            draw = ImageDraw.Draw(PIL_image)
            draw.polygon(xy=list(poly), outline=color, width=1)
            del draw 
        PIL_image=np.ascontiguousarray(PIL_image)
        save_image(PIL_image, 'test_video/output_images_det/', j)
        #存储跟踪框的图
        #在跟踪框上画上检测框
        frame = Image.fromarray(frame)
        for poly in pred_poly:
            color = (0,0,0)
            draw = ImageDraw.Draw(frame)
            draw.polygon(xy=list(poly), outline=color, width=1)
            del draw
        frame=np.ascontiguousarray(frame)   

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        save_image(frame, 'test_video/output_images/', j)


        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out_video.write(frame)

        # if c==27:
        #     capture.release()
        #     break

    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out_video.release()
    cv2.destroyAllWindows()