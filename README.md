# Yolov5 for Oriented Object Detection

The code for the implementation of “[Yolov5](https://github.com/ultralytics/yolov5) + [Circular Smooth Label](https://arxiv.org/abs/2003.05597v2)”. 

# Results and Models
The results on **DOTA_subsize1024_gap200_rate1.0** test-dev set are shown in the table below. (**password: yolo**)


<details>
  <summary>Table Notes (click to expand / **点我看更多**)</summary>

* All checkpoints are trained to 300 epochs with [COCO pre-trained checkpoints](https://github.com/ultralytics/yolov5/releases/tag/v6.0), default settings and hyperparameters.
* **mAP<sup>test dota</sup>** values are for single-model single-scale on [DOTA](https://captain-whu.github.io/DOTA/index.html)(1024,1024,200,1.0) dataset.<br>Reproduce Example:
 ```shell
 python val.py --data 'data/dotav15_poly.yaml' --img 1024 --conf 0.01 --iou 0.4 --task 'test' --batch 16 --save-json --name 'dotav15_test_split'
 python tools/TestJson2VocClassTxt.py --json_path 'runs/val/dotav15_test_split/best_obb_predictions.json' --save_path 'runs/val/dotav15_test_split/obb_predictions_Txt'
 python DOTA_devkit/ResultMerge_multi_process.py --scrpath 'runs/val/dotav15_test_split/obb_predictions_Txt' --dstpath 'runs/val/dotav15_test_split/obb_predictions_Txt_Merged'
 zip the poly format results files and submit it to https://captain-whu.github.io/DOTA/evaluation.html
 ```
* **Speed** averaged over DOTAv1.5 val_split_subsize1024_gap200 images using a 2080Ti gpu. NMS + pre-process times is included.<br>Reproduce by `python val.py --data 'data/dotav15_poly.yaml' --img 1024 --task speed --batch 1`


</details>

# [Updates](./docs/ChangeLog.md)
- [2022/1/7] : **Faster and stronger**, some bugs fixed, yolov5 base version updated.


# Installation
Please refer to [install.md](./docs/install.md) for installation and dataset preparation.

# Getting Started 
This repo is based on [yolov5](https://github.com/ultralytics/yolov5). 

And this repo has been rebuilt, Please see [GetStart.md](./docs/GetStart.md) for the Oriented Detection latest basic usage.

# train 
* 通过mk_train_list.py来制作训练和评估的数据集训练txt。
* python train.py   --weights yolov5s.pt   --data 'data/yolov5obb_demo.yaml'   --hyp 'data/hyps/obb/hyp.finetune_dota.yaml' --cfg models/yolov5s.yaml   --epochs 300   --batch-size 16   --img 640   --device 0

# val
* python val.py --data data/yolov5obb_demo.yaml  --weights /your weights_path/best.pt --task 'val'  --img 640 

# detect
* python detect.py --weights /your weights_path/best.pt   --source /test_img_path/   --img 640 --device 6 --conf-thres 0.25 --iou-thres 0.1 --hide-labels --hide-conf

# export
* python export.py --weights /your weights_path/best.pt  --batch 1

# -----剪枝---------
# 稀疏训练
* 先进行上面的预训练，对训练好的模型再进行稀疏训练
* python train_sparity.py --st --sr 0.0002 --weights /your weights_path/best.pt   --data data/yolov5obb_demo.yaml --epochs 100 --imgsz 640 --adam  --cfg models/yolov5s.yaml --batch-size 16

# 剪枝
* python prune.py --percent 0.7 --weights /your_save_path/last.pt --data data/yolov5obb_demo.yaml --cfg models/yolov5s.yaml

# 微调
* python prune_finetune.py --weights /your save_path/pruned_model.pt --data data/yolov5obb_demo.yaml  --epochs 100 --imgsz 640 --adam 

# 跟踪
* python track_predict.py

#  Acknowledgements
I have used utility functions from other wonderful open-source projects. Espeicially thank the authors of:

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
* [Thinklab-SJTU/CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow).
* [jbwang1997/OBBDetection](https://github.com/jbwang1997/OBBDetection)
* [CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
## More detailed explanation
想要了解相关实现的细节和原理可以看我的知乎文章:   
* [自己改建YOLOv5旋转目标的踩坑记录](https://www.zhihu.com/column/c_1358464959123390464).

## 有问题反馈
在使用中有任何问题，建议先按照[install.md](./docs/install.md)检查环境依赖项，再按照[GetStart.md](./docs/GetStart.md)检查使用流程是否正确，善用搜索引擎和github中的issue搜索框，可以极大程度上节省你的时间。



