# -*- coding: utf-8 -*-
import os
import random
trainval_percent = 1
train_percent = 1
xmlfilepath = '/labelTxt'
txtsavepath = 'images'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('trainval.txt', 'w')
ftest = open('/dataset//test.txt', 'w')
ftrain = open('dataset/train.txt', 'w')
fval = open('dataset/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
