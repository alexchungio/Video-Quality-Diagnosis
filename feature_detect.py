#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : feature_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/22 下午1:53
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import imageio
import imutils

image_path = './images/scene/scene_0.jpg'

train_image_path = './images/scene/scene_0.jpg'
query_image_path = './images/scene/scene_1.jpg'

def harris_corner_detect(image):
    rgb_img = cv.cvtColor(image, code=cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(image, code=cv.COLOR_BGR2GRAY)
    detect_dst = cv.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
    rgb_img[detect_dst > 0.005 * detect_dst.max()] = [0, 255, 0]
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()


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


def show_keypoint(train_img, train_kps, query_img, query_kps):
    """

    :param train_img:
    :param train_kps:
    :param query_img:
    :param query_kps:
    :return:
    """
    # display the keypoints and features detected on both images
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
    train_points_img = cv.drawKeypoints(train_img, train_kps, None, color=(0, 255, 0),
                                        flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    ax1.imshow(train_points_img)
    ax1.set_xlabel("(train)", fontsize=14)
    query_points_img = cv.drawKeypoints(query_img, query_kps, None, color=(0, 255, 0),
                     flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    ax2.imshow(query_points_img)
    ax2.set_xlabel("(query)", fontsize=14)
    plt.show()


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

    # compute the raw matches and initialize the list of actual matches
    match_point = matcher.knnMatch(features_a, features_b, 2)
    # print("Raw matches (knn):", len(match_point))
    matches = []
    # loop over the raw matches
    for m, n in match_point:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def get_homography(kps_a, kps_b, matches, reproj_thresh=4.0):
    # convert the keypoints to numpy arrays
    kps_a = np.float32([kp.pt for kp in kps_a])
    kps_b = np.float32([kp.pt for kp in kps_b])


    if len(matches) > 4:

        # construct the two sets of points
        points_a = np.float32([kps_a[m.queryIdx] for m in matches])
        points_b = np.float32([kps_b[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv.findHomography(points_a, points_b, cv.RANSAC, reproj_thresh)

        return (matches, H, status)
    else:
        return None



def main():
    # bgr_img = cv.imread(image_path, flags=cv.IMREAD_COLOR)
    #
    # harris_corner_detect(bgr_img)

    train_img = imageio.imread(train_image_path)
    train_img_gray = cv.cvtColor(train_img, cv.COLOR_RGB2GRAY)

    query_img = imageio.imread(query_image_path)
    # Opencv defines the color channel in the order BGR.
    # Transform it to RGB to be compatible to matplotlib
    query_img_gray = cv.cvtColor(query_img, cv.COLOR_RGB2GRAY)

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    # ax1.imshow(query_img_gray, cmap="gray")
    # ax1.set_xlabel("Query image", fontsize=14)
    #
    # ax2.imshow(train_img_gray, cmap="gray")
    # ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)
    # plt.show()

    feature_extractor = 'orb'
    ratio = 0.7
    kps_a, des_a = detector(train_img_gray, feature_extractor)
    kps_b, des_b = detector(query_img_gray, feature_extractor)

    # show_keypoint(train_img_gray, kps_a, query_img_gray, kps_b)

    matches = match_keypoint_with_knn(des_a, des_b, ratio=ratio, method=feature_extractor)
    print(len(matches))

    match_img = cv.drawMatches(train_img, kps_a, query_img, kps_b, matches,
                           None, matchColor=(0, 255, 0), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_img)
    plt.show()



if __name__ == "__main__":
    main()


