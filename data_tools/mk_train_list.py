import os

#将数据存放到训练和评估的txt文件中
path='yolov5_obb-master/dataset/'
with open('yolov5_obb-master/dataset/val.txt','w') as f:
    for filename in os.listdir(path):
        path1=path+'/'+filename
        for filename1 in os.listdir(path1):
            print(filename1)
            if filename1=='images':
                path2=path1+'/'+filename1
                for filename2 in os.listdir(path2):
                    path3=path2+'/'+filename2
                    if filename2=='val':
                        for filename3 in os.listdir(path3):
                            print(filename3)
                            f.write(path3+'/'+filename3+'\n')