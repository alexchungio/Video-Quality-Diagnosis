#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : gray_judge.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/1 下午2:40
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import cv2 as cv
import numpy as np
from tools import show_histogram

img_path = './images/demo.jpg'


def gray_judge(image, threshold=0.83, visual=False):

    shape = image.shape

    if len(shape) == 2:
        return True
    else:
        # convert bgr to hsv
        hsv_img = cv.cvtColor(image, code=cv.COLOR_RGB2HSV)
        hsv_hist = cv.calcHist([hsv_img], [1], None, [256], [0, 255])

        low_saturation_ratio = sum(hsv_hist[:25]) / sum(hsv_hist)

        if visual:
            show_histogram(hsv_img, 'hsv')

        return low_saturation_ratio[0] > threshold


def main():


    ratio = 0.001
    bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2GRAY)

    cv.imwrite('./images/gray_img.jpg', gray_img)
    #
    hsv_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2HSV)
    # bgr_hsv_hist = cv.calcHist([hsv_img], [1], None, [256], [0, 255])
    #
    # show_histogram(bgr_img, 'brg')
    # show_histogram(hsv_img, 'hsv')
    #
    # new_bgr_img = cv.cvtColor(gray_img, code=cv.COLOR_GRAY2BGR)
    new_bgr_img = np.stack([gray_img, gray_img, gray_img], axis=2)
    new_hsv_img = cv.cvtColor(new_bgr_img, code=cv.COLOR_BGR2HSV)
    # show_histogram(new_bgr_img, 'new_bgr')
    # show_histogram(new_hsv_img, 'new_hsv')
    #
    # gray_hsv_hist = cv.calcHist([new_hsv_img], [1], None, [256], [0,255])

    # tune saturation
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1].copy() * 0.2, 0, 255).astype(np.uint8)

    new_bgr_img = cv.cvtColor(hsv_img, code=cv.COLOR_HSV2BGR)

    # flag_0 = gray_detect(bgr_img)
    flag_1 = gray_judge(new_bgr_img)

    print(flag_1)

    cv.imshow('origin', bgr_img)
    cv.imshow('new', new_bgr_img)
    cv.waitKey(0)

    print('Done')


if __name__ == "__main__":
    main()