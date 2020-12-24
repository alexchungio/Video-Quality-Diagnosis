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
import os
import cv2 as cv

from jitter_detect import video_jitter_detect
from scene_change_detect import scene_detect
from brightness_abnormal import brightness
from joint_detect_blur_noise import detect_blur_noise
from color_deviation import color_deviation
from frozen_detect import frozen_detect
from gray_judge import gray_judge
from occlusion_detect import occlusion_detect
from contrast_abnormal import contrast_detect
from stripe_noise import detect_stripe


from tools import aspect_resize

jitter_video_path = './images/jitter/jitter.mp4'
stability_video_path = './images/jitter/stability.mp4'
scene_video_path = './images/scene/scene_1.mp4'

normal_image_path = './images/demo_1.jpg'
color_image_path = './images/color_deviation/deviation_0.jpg'


class VideoProperty(object):
    def __init__(self, min_size=256, jitter_threshold=3, scene_threshold=20):

        self.min_size = min_size
        self.jitter_threshold = jitter_threshold
        self.scene_threshold = scene_threshold
        self.image_property = ImageProperty()

    def __call__(self, cap, max_jitter_frame=20, max_change_frame=20, max_frozen_frame=25):

        video_quality = {}

        fps = int(cap.get(cv.CAP_PROP_FPS))
        self.jitter_frame = 0
        self.scene_change_frame = 0
        self.frozen_frame = 0
        _, pre_frame = cap.read()

        # detect video quality
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
                frozen = frozen_detect(pre_frame, cur_frame)
                # flag = 'jitter' if jitter else 'stability'
                # color = (0, 0, 255) if jitter else (0, 255, 0)
                # cv.putText(cur_frame, flag, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                #            0.7, color, 2)

                self.jitter_frame += int(jitter)
                self.scene_change_frame += int(scene_change)
                self.frozen_frame += int(frozen)
                pre_frame = cur_frame

            except Exception as e:
                print("Can't receive frame, due to {}".format(e))
                break

        video_quality['jitter'] = self.jitter_frame > max_jitter_frame
        video_quality['scene_change'] = self.scene_change_frame > max_change_frame
        video_quality['frozen'] = self.frozen_frame > max_frozen_frame

        # detect image quality
        video_quality = self.image_property(pre_frame, video_quality)

        return video_quality



class ImageProperty(object):
    def __init__(self, low_brightness_threshold=0.1, high_brightness_threshold=0.1, contrast_threshold=0.7,
                 gray_threshold=0.87, color_deviation_threshold=1.5, blur_threshold=20, noise_threshold=50,
                 occlusion_threshold_0=0.2, occlusion_threshold_1=0.25, strip_threshold=150):
        self.gray_threshold = gray_threshold
        self.low_brightness_threshold = low_brightness_threshold
        self.high_brightness_threshold = high_brightness_threshold
        self.contrast_threshold = contrast_threshold
        self.color_deviation_threshold = color_deviation_threshold
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold
        self.occlusion_threshold_0 = occlusion_threshold_0
        self.occlusion_threshold_1 = occlusion_threshold_1
        self.strip_threshold = strip_threshold

    def __call__(self, image, video_quality:dict = None):
        assert len(image.shape) == 3

        if video_quality is None:
            video_quality = {}

        gray_img = cv.cvtColor(image, code=cv.COLOR_BGR2GRAY)
        lab_img = cv.cvtColor(image, code=cv.COLOR_BGR2Lab)
        hsv_img = cv.cvtColor(image, code=cv.COLOR_BGR2HSV)
        gray_hist = cv.calcHist([gray_img], channels=[0], mask=None, histSize=[256],
                           ranges=[0, 255])
        video_quality['gray'] = gray_judge(image, threshold=self.gray_threshold)
        video_quality['low_brightness'], video_quality['high_brightness'] = brightness(gray_img,
                                                     low_threshold=self.low_brightness_threshold,
                                                     high_threshold=self.high_brightness_threshold,
                                                     gray_hist=gray_hist)
        video_quality['contrast'] = contrast_detect(gray_img, threshold=self.contrast_threshold, gray_hist=gray_hist)
        video_quality['color_deviation'] = color_deviation(lab_img, threshold=self.color_deviation_threshold)

        video_quality['blur'], video_quality['noise'] = detect_blur_noise(gray_img,
                                                  blur_threshold=self.blur_threshold,
                                                  noise_threshold=self.noise_threshold)

        video_quality['occlusion'] = occlusion_detect(gray_img, hsv_img,
                                          threshold_0=self.occlusion_threshold_0,
                                          threshold_1=self.occlusion_threshold_1)
        video_quality['strip'] = detect_stripe(hsv_img, threshold=self.strip_threshold)


        return video_quality


def main():
    video_property = VideoProperty()
    image_property = ImageProperty()

    cap = cv.VideoCapture(scene_video_path)
    #
    video_quality = video_property(cap)

    image = cv.imread(color_image_path, flags=cv.IMREAD_COLOR)
    image = aspect_resize(image)

    print(video_quality)

    # dataset = os.path.join('/media/alex/80CA308ECA308288/picture', 'dataset_1')
    #
    # image_name = os.listdir(dataset)
    #
    # for img_name in image_name[:15]:
    #
    #     print(img_name)
    #     img_path = os.path.join(dataset, img_name)
    #     image = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    #     image = aspect_resize(image)
    #
    #     print(image_property(image))



if __name__ == "__main__":
    main()