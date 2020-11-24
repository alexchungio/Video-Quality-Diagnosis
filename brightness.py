#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : brightness.py
# @ Description:  
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

img_path = './images/demo.jpg'

from enum import Enum

class Brightness(Enum):

    Normal = 0
    LOW = 1
    HIGH = 2


def brightness(bgr_img, low_threshold=0.1, high_threshold=0.1):

    flag = 0

    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([gray_img], channels=[0], mask=None, histSize=[256],
                       ranges=[0, 255])

    low_rate = sum(hist[:20]) / sum(hist)
    high_rate = sum(hist[-20:]) / sum(hist)

    if low_rate > low_threshold:
        flag = Brightness.LOW.value
    elif high_rate > high_threshold:
        flag = Brightness.HIGH.value

    return flag


def tune_contrast_with_lab(bgr_img, alpha=1.0):
    """

    :param bgr_img:
    :param factor:
    :return:
    """
    lab_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2Lab)
    lab_img = lab_img.astype(np.float32)
    lab_img[:, :, 0] *= alpha

    lab_img = np.clip(lab_img, a_min=0, a_max=255)
    lab_img = lab_img.astype(np.uint8)

    new_brg_img = cv.cvtColor(lab_img, cv.COLOR_Lab2LBGR)

    return new_brg_img

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


def tune_saturation_with_hsv(bgr_img, gamma=1.0):
    """

    :param bgr_img:
    :param factor:
    :return:
    """

    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    hsv_img = hsv_img.astype(np.float32)
    hsv_img[:, :, 1] *= gamma

    hsv_img = np.clip(hsv_img, a_min=0, a_max=255)
    hsv_img = hsv_img.astype(np.uint8)

    new_brg_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)

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



def show_histogram(image, title='hist'):
    """

    :param image:
    :param channel:
    :param title:
    :return:
    """

    # fig, axs = plt.subplots(2, 2)
    # # draw histogram

    # plt.hist(image.ravel(), bins=20)

    # display split channel
    img_channel = cv.split(image)
    colors = ['b', 'g', 'r']
    for color, channel in zip(colors, img_channel):
        sns.distplot(channel.ravel(), bins=10, kde=True, label=color)
    plt.title('RGB histogram')
    plt.legend()
    plt.show()

    # gray histogram
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sns.distplot(gray_image.ravel(), bins=10, kde=True)
    plt.title('GRY histogoram')
    plt.show()

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

    normal_flag = brightness(bgr_img)
    high_flag = brightness(high_brightness_img)
    low_flag = brightness(low_brightness_img)

    print(normal_flag)
    print(high_flag)
    print(low_flag)


    # cv.waitKey()
if __name__ == "__main__":
    main()

