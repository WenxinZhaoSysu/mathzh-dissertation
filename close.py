import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def coloring_segs(label_map, label_to_color=[]):
    img_c = np.zeros(label_map.shape, np.uint8)
    if len(label_to_color) == 0:
        label_to_color = np.unique(label_map)

    for l in label_to_color:
        mask = label_map == l
        img_c[mask] = 255
    return img_c
tests_path = glob.glob('data/test4/*.jpg')
for test_path in tests_path:
    save_res_path = test_path.split('.')[0] + '_after.png'
    img = cv2.imread(test_path.split('.')[0] + '_res.png')
    SE_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, SE_1)
    # plt.imshow(img1)
    imgg = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    num_label, labels, stats, centroids = cv2.connectedComponentsWithStats(imgg, connectivity=8)
    idx = np.argsort(stats[:, -1])
    img2 = coloring_segs(labels, idx[-2:-1])
    # img2=cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img2, 'gray')
    cv2.imwrite(save_res_path, img2)
