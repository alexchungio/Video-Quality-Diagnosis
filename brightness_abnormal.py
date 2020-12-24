#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : brightness_abnormal.py
# @ Description:  'https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html'
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/23 下午4:21
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import seaborn as sns
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tools import show_histogram

img_path = './images/demo_1.jpg'

from enum import Enum


class Brightness(Enum):

    Normal = 0
    LOW = 1
    HIGH = 2


def brightness(gray_img, low_threshold=0.1, high_threshold=0.1):


    assert len(gray_img.shape) == 2
    hist = cv.calcHist([gray_img], channels=[0], mask=None, histSize=[256],
                       ranges=[0, 255])

    low_rate = sum(hist[:20]) / sum(hist)
    high_rate = sum(hist[-20:]) / sum(hist)

    low_flag =  low_rate[0] > low_threshold
    high_flag = high_rate[0] > high_threshold


    return low_flag, high_flag


def tune_brightness_with_lab(bgr_img, beta=0):
    """

    :param bgr_img:
    :param factor:
    :return:
    """
    assert isinstance(beta, int)

    lab_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2Lab)
    lab_img[:, :, 0] += beta

    lab_img = np.clip(lab_img, a_min=0, a_max=255)

    new_brg_img = cv.cvtColor(lab_img, cv.COLOR_Lab2LBGR)

    return new_brg_img



def tune_contrast_brightness(image, alpha=1.0, beta=0):

    assert isinstance(beta, int)
    # Initialize values
    # try:
    #     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #     beta = int(input('* Enter the beta value [0-100]: '))
    # except ValueError:
    #     print('Error, not a number')

    # new_image = np.zeros(image.shape, image.dtype)

    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    new_image = image * alpha + beta
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    return new_image


def main():

    alpha = 1.2
    beta = 80
    gamma = 1.4
    bgr_img = cv.imread(img_path)
    # lab_contrast_img = tune_contrast_with_lab(bgr_img, alpha=alpha)
    # hsv_saturation_img = tune_saturation_with_hsv(bgr_img, gamma=gamma)
    # contrast_img = tune_contrast_brightness(bgr_img, alpha, 0)
    high_brightness_img = tune_contrast_brightness(bgr_img, 1.0, beta)
    low_brightness_img = tune_contrast_brightness(bgr_img, 1.0, -beta)

    # cv.imshow('raw image', bgr_img)
    # cv.imshow('lab contrast image', lab_contrast_img)
    # cv.imshow('hsv saturation image', hsv_saturation_img)
    # cv.imshow('contrast image', contrast_img)
    # cv.imshow('brightness image', brightness_img)

    # show_histogram(bgr_img, 'raw')
    # show_histogram(brightness_img, title='brightness')
    cv.imwrite('./images/hight_bright.jpg', high_brightness_img)
    cv.imwrite('./images/low_bright.jpg', low_brightness_img)

    high_brightness_img = cv.cvtColor(high_brightness_img, cv.COLOR_BGR2GRAY)
    low_brightness_img = cv.cvtColor(low_brightness_img, cv.COLOR_BGR2GRAY)
    high_flag = brightness(high_brightness_img)
    low_flag = brightness(low_brightness_img)


    print(high_flag)
    print(low_flag)

    # cv.waitKey()
if __name__ == "__main__":
    main()

