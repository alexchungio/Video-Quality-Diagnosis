#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : frozen_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/23 下午1:40
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import cv2 as cv
from tools import aspect_resize


stability_video_path = './images/jitter/stability.mp4'

def frozen_detect(pre_frame, cur_frame):

    return (pre_frame == cur_frame).all()


def main():
    cap = cv.VideoCapture(stability_video_path)

    fps = cap.get(cv.CAP_PROP_FPS)
    wait_time = int(1000 / fps)  # ms

    _, pre_frame = cap.read()
    pre_frame = aspect_resize(pre_frame)
    while cap.isOpened():

        try:
            start_time = time.perf_counter()
            ret, cur_frame = cap.read()
            if not ret:
                break

            cur_frame = aspect_resize(cur_frame)
            frozen = frozen_detect(pre_frame, cur_frame)
            flag = 'frozen' if frozen else 'normal'
            color = (0, 0, 255) if frozen else (0, 255, 0)
            cv.putText(cur_frame, flag, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
            pre_frame = cur_frame

            cost_time = int((time.perf_counter() - start_time) * 1000)  # ms
            cv.imshow('video', cur_frame)
            cv.waitKey(wait_time - cost_time)

        except Exception as e:
            print(e)
            print("Can't receive frame")


if __name__ == "__main__":
    main()


