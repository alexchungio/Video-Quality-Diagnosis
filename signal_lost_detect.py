#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : signal_lost_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/23 下午1:43
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
from tools import show_histogram, aspect_resize

image_path = './images/signal_lost/signal.jpg'



def lost_signal(image, threshold=0.98):

    hsv_img = cv.cvtColor(image, code=cv.COLOR_RGB2HSV)

    h_hist = cv.calcHist([hsv_img[:, :, 0]], channels=[0], mask=None, histSize=[256],
                       ranges=[0, 255])

    main_value = np.argmax(h_hist)

    main_percent = h_hist[main_value-2: main_value+2].sum() / h_hist.sum()

    return main_percent > threshold


def main():

    bgr_img = cv.imread(image_path, flags=cv.IMREAD_COLOR)
    bgr_img = aspect_resize(bgr_img)

    print(lost_signal(bgr_img))

    print('Done')


if __name__ == "__main__":
    main()

