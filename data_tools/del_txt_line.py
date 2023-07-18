import os

path=r'/home/yuanzhengqian/yolov5_obb-master/dataset/DOTA/val/images/labelTxt/'
i=0
for filename in os.listdir(path):
    txt_path=path+filename

    # txt文本单次删除第一行
    with open(txt_path, mode='r', encoding='utf-8') as f:
        line = f.readlines()  # 读取文件
        try:
            line = line[2:]  # 只读取第一行之后的内容
            f = open(txt_path, mode='w', encoding='utf-8')  # 以写入的形式打开txt文件
            f.writelines(line)    # 将修改后的文本内容写入
            f.close()             # 关闭文件
        except:
            pass