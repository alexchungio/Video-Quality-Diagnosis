#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : scene_change_detect.py
# @ Description: https://docs.opencv.org/4.4.0/db/d27/tutorial_py_table_of_contents_feature2d.html
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/17 上午9:30
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import cv2 as cv


image_path = './images/cat.jpg'
video_path = './images/scene/scene_1.mp4'
output_path_0 = './outputs/output.avi'


def get_video_format(cap):
    """
    get video format
    """
    raw_codec_format = int(cap.get(cv.CAP_PROP_FOURCC))
    decoded_codec_format = (chr(raw_codec_format & 0xFF), chr((raw_codec_format & 0xFF00) >> 8),
                            chr((raw_codec_format & 0xFF0000) >> 16), chr((raw_codec_format & 0xFF000000) >> 24))
    return decoded_codec_format


def scene_detect(train_img, query_img, extractor_method='orb', match_type='flann', ratio=0.8, threshold=10):
    """

    :param pre_img:
    :param cur_img:
    :param method:
    :param match_type:
    :param ratio:
    :return:
    """

    # gray image
    train_img_gray = cv.cvtColor(train_img, cv.COLOR_RGB2GRAY)
    query_img_gray = cv.cvtColor(query_img, cv.COLOR_RGB2GRAY)

    # detect
    kps_a, des_a = detector(train_img_gray, extractor_method)
    kps_b, des_b = detector(query_img_gray, extractor_method)

    matches = match_keypoint_with_knn(des_a, des_b, ratio=ratio, match_type=match_type, method=extractor_method)

    print(len(matches))
    return len(matches) < threshold


def detector(image, method='sift'):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        detector = cv.SIFT_create()
    # elif method == 'surf':
    #     detector = cv.xfeatures2d.SURF_create()
    elif method == 'fast':
        detector = cv.FastFeatureDetector_create()
    elif method == 'brisk':
        detector = cv.BRISK_create()
    elif method == 'orb':
        detector = cv.ORB_create()
    elif method == 'akaze':
        detector = cv.AKAZE_create()
    else:
        raise KeyError('method must between in sift surf fast brisk orb and akaze')

    # get keypoints and descriptors
    (kps, des) = detector.detectAndCompute(image, None)

    return (kps, des)



def create_matcher_with_bf(method=None, cross_check=True):
    "Create and return a Matcher Object"
    if method == 'sift' or method == 'surf':
        matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=cross_check)
    elif method == 'orb' or method == 'brisk' or method == 'akaze':
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=cross_check)

    return matcher


def create_matcher_with_flann(checks=50):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=checks)
    matcher = cv.FlannBasedMatcher(index_params, search_params)

    return matcher


def match_keypoint(features_a, features_b, match_type='bf', method='sift'):

    if match_type == 'bf':
        matcher = create_matcher_with_bf(method, cross_check=True)
    elif match_type == 'flann':
        matcher = create_matcher_with_flann()

    # Match descriptors.
    best_matches = matcher.match(features_a, features_b)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    sorted_match = sorted(best_matches, key=lambda x: x.distance)
    # print("Raw matches (Brute force):", len(sorted_match))

    return sorted_match


def match_keypoint_with_knn(features_a, features_b, ratio=0.7, match_type='bf', method='sift'):
    if match_type == 'bf':
        matcher = create_matcher_with_bf(method, cross_check=False)
    elif match_type == 'flann':
        matcher = create_matcher_with_flann()

        # flann need convert to float32
        features_a = features_a.astype(np.float32)
        features_b = features_b.astype(np.float32)

    # compute the raw matches and initialize the list of actual matches
    match_point = matcher.knnMatch(features_a, features_b, k=2)
    # print("Raw matches (knn):", len(match_point))
    matches = []
    # loop over the raw matches
    for m, n in match_point:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


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
    _, pre_frame = cap.read()
    while cap.isOpened():
        try:
            ret, cur_frame = cap.read()
            if not ret:
                break

            is_change = scene_detect(cur_frame, pre_frame)
            flag = 'change' if is_change else 'same'
            color = (0, 0, 255) if is_change else (0, 255, 0)
            cv.putText(cur_frame, flag, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
            out_writer.write(cur_frame)
            cv.imshow('video', cur_frame)
            cv.waitKey(wait_time)
        except:
            print("Can't receive frame")


if __name__ == "__main__":
    main()