#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/24 上午11:32
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt


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
    assert len(image.shape) in [2, 3]
    if image.shape == 3:
        # display split channel
        img_channel = cv.split(image)
        colors = ['b', 'g', 'r']
        for color, channel in zip(colors, img_channel):
            sns.distplot(channel.ravel(), bins=25, kde=True, label=color)
        plt.title('RGB histogram')
        plt.legend()
        plt.show()

        # gray histogram
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sns.distplot(gray_image.ravel(), bins=25, kde=True)
        plt.title('GRY histogram')
        plt.show()
    else:
        sns.distplot(image.ravel(), bins=25, kde=True)
        plt.title('GRY histogram')
        plt.show()


def visual_fft_spectrum(fft, title='fft_spectrum'):
    """

    :param spectrum:
    :param title:
    :return:
    """
    # centralize
    central_f = np.fft.fftshift(fft)
    # real value
    abs_f = np.abs(central_f)
    # change scale
    scale_f = np.log(1 + abs_f)

    # convert scale to (0, 255)
    min = np.min(scale_f)
    max = np.max(scale_f)
    scale_f = -min + 255 / (max - min) * scale_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()


def visual_fft_phase(fft, title='fft_phase'):
    """

    :param fft:
    :param title:
    :return:
    """
    # centralize
    central_f = np.fft.fftshift(fft)
    real_f = np.real(central_f)
    imag_f = np.imag(central_f)
    phase_f = np.arctan2(imag_f, real_f)
    # convert scale to (0, 255)
    min = np.min(phase_f)
    max = np.max(phase_f)
    scale_f = -min + 255 / (max - min) * phase_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()