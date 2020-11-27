#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : color_deviation.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/27 上午9:50
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np


img_path = './images/cat.jpg'

def tune_color(image, alpha=1.8, beta=1.0, gamma=1.0):

    # lab_img = cv.cvtColor(image, code=cv.COLOR_BGR2Lab)

    # convert float
    bgr_img = image.astype(np.float32)

    b, g, r = cv.split(bgr_img)

    b *= alpha
    g *= beta
    r *= gamma

    bgr_img = cv.merge([b, g, r])

    bgr_img = np.clip(bgr_img, 0, 255).astype(np.uint8)
    # bgr_img = cv.cvtColor(lab_img, code=cv.COLOR_Lab2LBGR)
    return bgr_img


def color_deviation(image, threshold = 1.5):
    """

    :param image:
    :param threshold:
    :return:
    """
    lab_img = cv.cvtColor(image, code=cv.COLOR_BGR2Lab)

    lab_img = lab_img.astype(np.float32)

    l, a, b = cv.split(lab_img)

    d_a = np.mean(a) - 128
    d_b = np.mean(b) - 128
    d = np.sqrt(d_a ** 2 + d_b **2)

    m_a = np.mean(np.abs(a - d_a - 128))
    m_b = np.mean(np.abs(b - d_b - 128))
    m = np.sqrt(m_a ** 2 + m_b ** 2)

    k = d / m

    return k < threshold


def main():
    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    color_img = tune_color(image)
    origin_k = color_deviation(image)
    color_k = color_deviation(color_img)
    print(origin_k)
    print(color_k)
    cv.imshow('origin image', image)
    cv.imshow('color img', color_img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()