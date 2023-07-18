# -*- coding: utf-8 -*-
 
import os
import random
 
# obb data split
annfilepath=r'dataset/labelTxt'
saveBasePath=r'dataset'
train_percent=1.0
total_file = os.listdir(annfilepath)
num=len(total_file)
list=range(num)
tr=int(num*train_percent)
train=random.sample(list,tr)
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')
for i  in list:
    name=total_file[i].split('.')[0]+'\n'
    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)
ftrain.close()
fval.close()
print("train size",tr)
print("valid size",num-tr)