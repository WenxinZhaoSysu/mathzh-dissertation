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
            elif(y_true[i][j] == y_pred[i][j])&(y_pred[i][j]==255):
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


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    # net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.load_state_dict(torch.load('best_model_se80.pth', map_location=device))

    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.jpg')
    ious = []
    pe = []
    ri = []
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        true_path=test_path.replace('data/test4', "all_label").replace('.jpg', '_segmentation.png')
        true=cv2.imread(true_path)
        true = cv2.cvtColor(true, cv2.COLOR_RGB2GRAY)
        # true= np.double(cv2.resize(img, (600,450)))
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img= np.double(cv2.resize(img, (256, 256)))
        # 转为batch为1，通道为1，大小为512*512的数组 128
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # pred=cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        pred = np.double(cv2.resize(pred, (600, 450)))
        cv2.imwrite(save_res_path, pred)
        print(pred,true)
        ious.append(iou(true,pred))
        print(ious[-1])
        pe.append(pixel_error(true,pred))
        print(pe[-1])
        ri.append(rand_index(true,pred))
        print(ri[-1])
        # 保存图片
        cv2.imwrite(save_res_path, pred)

    # print(pe.average,ri.average,iou.average)


