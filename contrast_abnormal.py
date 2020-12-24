#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : contrast_abnormal.py
# @ Description: https://eurradiolexp.springeropen.com/track/pdf/10.1186/s41747-017-0023-4.pdf
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/24 上午11:36
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
from tools import show_histogram

img_path = './images/fog_1.jpg'


def contrast_detect(gray_img, threshold=0.7, range=40, gray_hist=None):
    """

    :param gray_img:
    :param threshold:
    :param range:
    :param gray_hist:
    :return:
    """

    assert len(gray_img.shape) == 2
    median_size = np.median(gray_img)

    if gray_hist is not None:
        hist = gray_hist.flatten()
    else:
        hist = cv.calcHist([gray_img], channels=[0], mask=None, histSize=[256],
                           ranges=[0, 255]).flatten()

    low_threshold, high_threshold = np.clip((median_size-range, median_size+range), 0, 255).astype(np.int32)

    gather_pixel = hist[low_threshold:high_threshold]

    gather_rate = sum(gather_pixel) / sum(hist)
    # print(gather_rate)

    return gather_rate > threshold



def low_contrast(src, fraction_threshold=0.5, lower_percentile=1,
                    upper_percentile=99):
    """

    :param src:
    :param fraction_threshold:
    :param lower_percentile:
    :param upper_percentile:
    :return:
    """
    # convert to gray image
    src = np.asanyarray(src)

    if src.ndim == 3 and src.shape[2] in [3, 4]:
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    # get min and max pixel
    dlimits = (np.iinfo(src.dtype).min, np.iinfo(src.dtype).max)
    # computer low pixel percent
    limits = np.percentile(src, [lower_percentile, upper_percentile])
    # computer ratio
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    return ratio < fraction_threshold


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

    bgr_img = cv.imread(img_path)
    # contrast_img = tune_contrast_brightness(bgr_img, alpha)
    # gray_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2GRAY)
    # show_histogram(bgr_img)
    # cv.imshow('raw image', bgr_img)
    # cv.waitKey(0)

    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    contrast_flag = contrast_detect(gray_img)
    # contrast_flag = low_contrast(bgr_img)
    print(contrast_flag)

if __name__ == "__main__":
    main()