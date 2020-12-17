#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : scene_change_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/17 上午9:30
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2 as cv


image_path = './images/cat.jpg'
video_path = './images/scene/scene_0.mp4'
output_path_0 = './outputs/output.avi'


def get_video_format(cap):
    """
    get video format
    """
    raw_codec_format = int(cap.get(cv.CAP_PROP_FOURCC))
    decoded_codec_format = (chr(raw_codec_format & 0xFF), chr((raw_codec_format & 0xFF00) >> 8),
                            chr((raw_codec_format & 0xFF0000) >> 16), chr((raw_codec_format & 0xFF000000) >> 24))
    return decoded_codec_format


def main():

    # bgr_img = cv.imread(image_path, flags=cv.IMREAD_COLOR)
    # cv.imshow('raw image', bgr_img)
    # cv.waitKey(0)

    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    wait_time = int(1000 / fps)  # ms

    # https://github.com/skvark/opencv-python/issues/100
    # use openh264
    fourcc = cv.VideoWriter_fourcc('X','V', 'I', 'D')
    out_writer = cv.VideoWriter(filename=output_path_0, fourcc=fourcc, fps=fps, frameSize=(width, height),
                         isColor=True)
    while cap.isOpened():
        try:
            ret, cur_frame = cap.read()
            if not ret:
                break
            out_writer.write(cur_frame)
            cv.imshow('video', cur_frame)
            cv.waitKey(wait_time)
        except:
            print("Can't receive frame")


if __name__ == "__main__":
    main()