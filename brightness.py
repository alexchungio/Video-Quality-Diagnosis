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
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_path = './images/demo.jpg'

def brightness(img, low_threshold, high_threshold):
    pass



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

def tune_saturation_with_hsv(bgr_img, gamma=0):
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



def show_histogram(image, channel, title='hist'):
    """

    :param image:
    :param channel:
    :param title:
    :return:
    """
    img_channel = cv.split(image)
    hist = cv.calcHist([img_channel[channel]], channels=[channel], mask=None, histSize=[256],
                       ranges=[0, 256])
    plt.plot(hist, color=title)
    plt.xlim([0, 256])

def main():

    alpha = 1.2
    beta = 20
    gamma = 1.4
    bgr_img = cv.imread(img_path)
    lab_contrast_img = tune_contrast_with_lab(bgr_img, alpha=alpha)
    hsv_saturation_img = tune_saturation_with_hsv(bgr_img, gamma=gamma)
    contrast_img = tune_contrast_brightness(bgr_img, alpha, 0)
    brightness_img = tune_contrast_brightness(bgr_img, 1.0, beta)
    # chans = cv.split(Overexpose)
    # colors = ("b", "g", "r")
    # for (chan, color) in zip(chans, colors):
    #     hist = cv.calcHist([chan], [0], None, [256], [0, 256])
    #     plt.plot(hist, color=color)
    #     plt.xlim([0, 256])
    cv.imshow('raw image', bgr_img)
    cv.imshow('lab contrast image', lab_contrast_img)
    cv.imshow('hsv saturation image', hsv_saturation_img)
    cv.imshow('contrast image', contrast_img)
    cv.imshow('brightness image', brightness_img)
    cv.waitKey()

    show_histogram(brightness_img)
    plt.imshow()

if __name__ == "__main__":
    main()