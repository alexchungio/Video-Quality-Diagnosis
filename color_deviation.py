#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : color_deviation.py
# @ Description: reference
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/27 上午9:50
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
from tools import get_space_scale

img_path = './images/demo_1.jpg'

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


def color_deviation(lab_img, threshold = 1.5):
    """
    true scale vs opencv scale:
    l => (0, 100) => (0, 255)
    a => (-128, 128) =. (0, 255)
    b => (-128, 128) => (0, 255)
    :param image:
    :param threshold:
    :return:
    """
    assert len(lab_img.shape) == 3

    lab_img = lab_img.astype(np.float32)

    l, a, b = cv.split(lab_img)

    # convert scale to (-128, 128)
    # center coordinates of equivalent c
    d_a = np.mean(a) - 128   # (0, 255) => (-180, 180)
    d_b = np.mean(b) - 128  # (0, 255) => (-180, 180)
    # computer
    d = np.sqrt(d_a ** 2 + d_b **2)

    #
    # use abs replace square operation
    # convert scale to (-128, 128)
    m_a = np.mean(np.abs(a - d_a - 128))  # (0, 255) => (-180, 180)
    m_b = np.mean(np.abs(b - d_b - 128))  # (0, 255) => (-180, 180)
    m = np.sqrt(m_a ** 2 + m_b ** 2)

    k = d / m

    return k > threshold

def main():
    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    color_img = tune_color(image)

    origin_lab_img = cv.cvtColor(image, code=cv.COLOR_BGR2Lab)
    color_lab_img = cv.cvtColor(color_img, code=cv.COLOR_BGR2Lab)
    origin_k = color_deviation(origin_lab_img)
    color_k = color_deviation(color_lab_img)
    cv.imwrite('./images/color_deviation.jpg', color_img)
    # get_space_scale()
    print(origin_k)
    print(color_k)
    cv.imshow('origin image', image)
    cv.imshow('color img', color_img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()