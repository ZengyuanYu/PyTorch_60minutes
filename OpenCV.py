#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-2-26
import cv2
import numpy as np
from matplotlib import pyplot as plt

####################################
# 将图像的BGR顺序纠正为RGB
def bgr2rgb(src):
    img = src.copy()
    img[:,:,0] = src[:,:,2]
    img[:,:,2] = src[:,:,0]
    return img
######################################

print("图像平滑")

img = cv2.imread("juemingzi.jpg")

# 创建掩膜
k = 9
kernel = np.ones((k,k), np.float32)/k**2
print (kernel)

# 均值滤波
#cv.Filter2D(src, dst, kernel, anchor=(-1, -1))
#ddepth –desired depth of the destination image;
#if it is negative, it will be the same as src.depth();
#the following combinations of src.depth() and ddepth are supported:
#src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
#src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
#src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
#src.depth() = CV_64F, ddepth = -1/CV_64F
#when ddepth = -1, the output image will have the same depth as the source.
img_average = cv2.filter2D(img,-1,kernel)

# 使用blur()函数进行均值滤波
img_blur = cv2.blur(img, (k,k))

# 高斯平滑
img_gaussian = cv2.GaussianBlur(img, (k,k), 3)
# 也可以自己创建一个高斯核
#kernel_g = cv2.getGaussianKernel(k, 5)
#img_gaussian = cv2.filter2D(img,-1,kernel_g)
#print kernel_g.shape

# 中值滤波
#img_median = cv2.medianBlur(img_noise,k)

#plt.subplot(121),plt.imshow(img_noise,'gray'),plt.title("噪声图像")
#plt.subplot(122),plt.imshow(img_median,'gray'),plt.title("中值滤波后的图像")
#plt.show()

# 双边滤波
#cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
#d – Diameter of each pixel neighborhood that is used during filtering.
# If it is non-positive, it is computed from sigmaSpace
#9 邻域直径，两个75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
img_new = cv2.imread("Airplane.jpg")
img_bf = cv2.bilateralFilter(img_new,9,75,75)

# 平滑滤波结果
plt.subplot(221),plt.imshow(bgr2rgb(img),"gray"),plt.title("Original")
plt.xticks([]),plt.yticks([])  # 去掉坐标轴刻度
plt.subplot(222),plt.imshow(bgr2rgb(img_average),"gray"),plt.title("average_filtering")
plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(bgr2rgb(img_blur),"gray"),plt.title("blur_filtering")
plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(bgr2rgb(img_gaussian),"gray"),plt.title("gaussian_filtering")
plt.xticks([]),plt.yticks([])
plt.show()

# 双边滤波结果
plt.subplot(121),plt.imshow(bgr2rgb(img_new), "gray"), plt.title("Original")
plt.xticks([]),plt.yticks([])
plt.subplot(122), plt.imshow(bgr2rgb(img_bf), "gray"), plt.title("bilateralFilter")
plt.xticks([]), plt.yticks([])
plt.show()