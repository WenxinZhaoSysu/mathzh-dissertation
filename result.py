import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

def pixel_error(y_true, y_pred):
    n = y_true.shape[0]
    m= y_true.shape[1]
    error=y_true-y_pred
    dif=np.count_nonzero(error)
    pixel_error=dif/(n*m)
    # print(pixel_error)
    return pixel_error


def rand_index(y_true, y_pred):
    n = y_true.shape[0]
    m= y_true.shape[1]
    a, b = 0, 0
    for i in range(n):
        for j in range(m):
            if (y_true[i][j] == y_pred[i][j])&(y_pred[i][j]==0):
                a +=1
            elif (y_true[i][j] == y_pred[i][j])&(y_pred[i][j]==255):
                b +=1
            else:
                pass
    # print(a,b,n,m)
    RI = (a + b) /(n*m)
    return RI

def iou(y_true, y_pred):
    n = y_true.shape[0]
    m= y_true.shape[1]
    a, b = 0, 0
    for i in range(n):
        for j in range(m):
            if (y_true[i][j] != y_pred[i][j]):
                a +=1
            elif (y_true[i][j] == y_pred[i][j])&(y_pred[i][j]==255):
                b +=1
            else:
                pass
    print(a,b,n,m)
    iou = b/(a+b)
    return iou

ious=[]
pe=[]
ri=[]
tests_path = glob.glob('data/test4/*.jpg')
for test_path in tests_path:
    save_res_path = test_path.split('.')[0] + '_res.png'
    img = cv2.imread(test_path)
    true_path = test_path.replace('data/test4', "all_label").replace('.jpg', '_segmentation.png')
    true = cv2.imread(true_path)
    true = cv2.cvtColor(true, cv2.COLOR_RGB2GRAY)
    pred = cv2.imread(save_res_path)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    ious.append(iou(true, pred))
    print(ious[-1])
    pe.append(pixel_error(true, pred))
    print(pe[-1])
    ri.append(rand_index(true, pred))
    print(ri[-1])

print("pe",np.mean(pe))
print("ri",np.mean(ri))
print("iou:",np.mean(ious))