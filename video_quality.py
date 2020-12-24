#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : video_quality.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/24 上午10:26
# @ Software   : PyCharm
#-------------------------------------------------------

import time
import cv2 as cv

from jitter_detect import video_jitter_detect
from scene_change_detect import scene_detect
from tools import aspect_resize

jitter_video_path = './images/jitter/jitter.mp4'
stability_video_path = './images/jitter/stability.mp4'
scene_video_path = './images/scene/scene_1.mp4'

class VideoProperty(object):
    def __init__(self, min_size=256, jitter_threshold=3, scene_threshold=20):

        self.min_size = min_size
        self.jitter_threshold = jitter_threshold
        self.scene_threshold = scene_threshold

    def __call__(self, cap, max_jitter_frame=20, max_change_frame=20):

        fps = int(cap.get(cv.CAP_PROP_FPS))
        # wait_time = int(1000 / fps)  # ms

        self.jitter_frame = 0
        self.scene_change_frame = 0
        _, pre_frame = cap.read()
        for _ in range(fps):
        # while cap.isOpened():
            try:
                start_time = time.perf_counter()
                ret, cur_frame = cap.read()

                pre_frame = aspect_resize(pre_frame, self.min_size)
                cur_frame = aspect_resize(cur_frame, self.min_size)
                if not ret:
                    break
                jitter = video_jitter_detect(pre_frame, cur_frame, threshold=self.jitter_threshold)
                scene_change = scene_detect(pre_frame, cur_frame, threshold=self.scene_threshold)
                flag = 'jitter' if jitter else 'stability'
                color = (0, 0, 255) if jitter else (0, 255, 0)
                cv.putText(cur_frame, flag, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2)

                # cost_time = int((time.perf_counter() - start_time) * 1000)  # ms
                # cv.imshow('video', cur_frame)
                #
                # cv.waitKey(wait_time-cost_time)

                self.jitter_frame += int(jitter)
                self.scene_change_frame += int(scene_change)
                pre_frame = cur_frame
                # cost_time = int((time.perf_counter() - start_time) * 1000)  # ms
                # cv.waitKey(wait_time-cost_time)

            except Exception as e:
                print("Can't receive frame, due to {}".format(e))
                break

        jitter_flag = self.jitter_frame > max_jitter_frame
        scene_change_flag = self.scene_change_frame > max_change_frame

        return jitter_flag, scene_change_flag



class ImageProperty(object):
    def __init__(self):
        pass

    def __call__(self):
        pass


def main():
    video_property = VideoProperty()

    cap = cv.VideoCapture(jitter_video_path)

    jitter_flag, scene_change_flag = video_property(cap)

    print(jitter_flag, scene_change_flag)


if __name__ == "__main__":
    main()