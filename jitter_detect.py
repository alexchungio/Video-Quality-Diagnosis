#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : jitter_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/26 上午10:47
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_path = './images/cat.jpg'
img_path_2 = './images/demo.jpg'

jitter_video_path = './images/jitter/jitter.mp4'
stability_video_path = './images/jitter/stability.mp4'


def video_jitter_detect(pre_img, cur_img, threshold=3, m=3):

    pre_gray_img = cv.cvtColor(pre_img, code=cv.COLOR_BGR2GRAY)
    cur_gray_img = cv.cvtColor(cur_img, code=cv.COLOR_BGR2GRAY)


    pre_p_row, pre_p_col = gray_projection(pre_gray_img, reduce_mean=True)
    cur_p_row, cur_p_col = gray_projection(cur_gray_img, reduce_mean=True)


    delta_row = cross_correlation(pre_p_row, cur_p_row, m=m)
    delta_col = cross_correlation(pre_p_col, cur_p_col, m=m)

    print(delta_row, delta_col)
    is_jitter = abs(delta_row) >= threshold or abs(delta_col) >= threshold

    return is_jitter


def cross_correlation(ref_projection, cur_projection, m=4):
    """

    :param pre_projection: reference image projection
    :param cur_projection: current image projection
    :param m: search size
    :return:
    """
    cross_corr = []
    ref_projection = np.asarray(ref_projection)
    cur_projection = np.asarray(cur_projection)
    valid_length = len(ref_projection) - 2 * m
    ref_window = ref_projection[m: -m]
    for w in range(0, 2*m + 1):
        cur_window = cur_projection[w: w+valid_length]
        diff = (cur_window - ref_window) ** 2
        cross_corr.append(diff.sum())

    min_w = np.argmin(np.asarray(cross_corr))

    delta_moving = m - min_w

    return delta_moving


def plot_projection(p_row, p_col):

    row_x = np.arange(len(p_row))
    row_y = p_row

    col_x = np.arange(len(p_col))
    col_y = p_col

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    ax0.set_title('row dim gray projection')
    ax0.plot(row_x, row_y)
    ax1.set_title('col dim gray projection')
    ax1.plot(col_x, col_y)
    plt.show()


def gray_projection(gray_img, reduce_mean=False):

    g_row = gray_img.sum(axis=1)
    g_col = gray_img.sum(axis=0)

    # projection
    if reduce_mean:
        p_row = g_row - g_row.mean()
        p_col = g_col - g_col.mean()
    else:
        p_row = g_row
        p_col = g_col

    return p_row, p_col

def main():

    # brg_img = cv.imread(img_path)
    #
    # gray_img = cv.cvtColor(brg_img, code=cv.COLOR_BGR2GRAY)
    #
    # gray_projection(gray_img)
    #
    # p_row, p_col = gray_projection(gray_img, reduce_mean=True)
    #
    #
    # plot_projection(p_row, p_col)
    #
    # delta_x = cross_correlation(p_row, p_row, m=2)
    # print(delta_x)
    #
    # delta_y = cross_correlation(p_col, p_col, m=2)
    # print(delta_y)

    cap = cv.VideoCapture(stability_video_path)

    fps = cap.get(cv.CAP_PROP_FPS)
    wait_time = int(1000 / fps)

    _, pre_frame = cap.read()
    while cap.isOpened():
        try:
            ret, cur_frame = cap.read()
            if not ret:
                break

            jitter = video_jitter_detect(pre_frame, cur_frame)
            flag = 'jitter' if jitter else 'stability'
            color = (0, 0, 255) if jitter else (0, 255, 0)
            cv.putText(cur_frame, flag, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
            pre_frame = cur_frame
            cv.imshow('video', cur_frame)
            cv.waitKey(wait_time)
        except:
            print("Can't receive frame")


if __name__ == "__main__":
    # 'https://python.hotexamples.com/examples/cv2/-/phaseCorrelate/python-phasecorrelate-function-examples.html'
    # 'https://blog.csdn.net/bemy1008/article/details/86687793'
    main()