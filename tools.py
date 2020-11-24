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