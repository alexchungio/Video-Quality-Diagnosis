#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : saturation.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/24 上午10:21
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import cv2 as cv

img_path = './images/demo.jpg'


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


def tune_hue_saturation_lightness(brg_image, alpha=1.0, beta=1.0, gamma=1.0):
    """
    h => (0, 180) s =>(0, 255) l => (0, 255)
    :param brg_image:
    :param alpha:
    :param beta:
    :return:
    """


    hls_image = cv.cvtColor(brg_image, cv.COLOR_BGR2HLS)

    h_channel, l_channel, s_channel = cv.split(hls_image)

    # tune saturation
    h_channel = h_channel.astype(np.float32)
    h_channel *= alpha
    h_channel = np.clip(h_channel, 0, 180).astype(np.uint8)

    # tune saturation
    s_channel = s_channel.astype(np.float32)
    s_channel *= beta
    s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)

    # tune lightness
    l_channel = l_channel.astype(np.float32)
    l_channel *= gamma
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)

    new_hls_image = cv.merge([h_channel, l_channel, s_channel])

    new_bgr_image = cv.cvtColor(new_hls_image, cv.COLOR_HLS2BGR)

    return new_bgr_image


def main():
    alpha = 1.5
    beta = 1.0
    gamma = 1.0

    bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    lightness_img = tune_hue_saturation_lightness(bgr_img, alpha, beta, gamma)
    cv.imshow('lightness', lightness_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()