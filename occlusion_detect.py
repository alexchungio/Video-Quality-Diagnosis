#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : occlusion_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/7 下午3:21
# @ Software   : PyCharm
#-------------------------------------------------------


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img_path = './images/occlusion/occlusion_0.jpg'


def get_leaf_area(image, visualize=False):

    hsv_img = cv.cvtColor(image, code=cv.COLOR_RGB2HSV)
    low_threshold = (45, 45, 5)
    upper_threshold = (255, 255, 255)

    h_channel = hsv_img[:, :, 0]

    # adaptive_binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)

    # use h channels
    h_mask = (h_channel > 40) & (h_channel < 90)

    # use unit hsv space
    hsv_mask = cv.inRange(hsv_img, lowerb=low_threshold, upperb=upper_threshold) / 255

    tree_mask = np.asarray(h_mask * hsv_mask * 255, dtype=np.uint8)

    # morphology process
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # tree_mask = cv.morphologyEx(tree_mask, cv.MORPH_OPEN, kernel)
    # tree_mask = cv.morphologyEx(tree_mask, cv.MORPH_CLOSE, kernel, iterations=3)
    tree_mask = cv.dilate(tree_mask, kernel=kernel, iterations=3)


    # Find contours

    contours, hierarchy = cv.findContours(tree_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours

    # Show in a window


    if visualize:

        tree_area = cv.bitwise_and(image, image, mask=tree_mask)
        plt.imshow(tree_area[:, :, ::-1])
        plt.axis('off')
        plt.show()
        cv.drawContours(tree_area, contours, -1, (0, 0, 255), 3)
        plt.imshow(tree_area[:, :, ::-1])
        plt.show()


    return tree_mask

plt.ion()
def main():

    for i in range(3):
        img_path = './images/occlusion/occlusion_{}.jpg'.format(i)
        bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)

        tree_mask = get_leaf_area(bgr_img, visualize=True)

    print('Done !')


if __name__ == "__main__":
    main()