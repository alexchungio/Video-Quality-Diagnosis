#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : occlusion_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/7 下午3:21
# @ Software   : PyCharm
#-------------------------------------------------------


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as cocomask

# img_path = './images/occlusion/occlusion_1.jpg'
img_path = './images/cat.jpg'


def caustics_detect(gray_img, k=1):
    """

    :param image:
    :param k:
    :return:
    """
    h, w = gray_img.shape

    gradual_matrix = np.zeros((h, w))


def check_contour_include(large_contour, small_contour):
    """

    :param contour_0:
    :param contour_1:
    :return:
    """
    large_x, large_y = large_contour[:, :, 0].flatten(), large_contour[:, :, 1].flatten()
    small_x, small_y = small_contour[:, :, 0].flatten(), small_contour[:, :, 1].flatten()

    large_x = sorted(large_x)
    large_y = sorted(large_y)

    small_x = sorted(small_x)
    small_y = sorted(small_y)

    if small_x[0] > large_x[0] and small_x[-1] < large_x[-1]:
        if small_y[0]> large_y[0] and small_y[-1] < large_y[-1]:
            return True
        else:
            return False
    else:
        return False


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def find_contour(image):

    height, width = image.shape
    # Find contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_area = [cv.contourArea(contour) for contour in contours]

    area_index = [i for i, v in sorted(enumerate(contour_area), key=lambda x: x[1])[::-1]][:3]

    top_contour = [contours[index] for index in area_index]

    mask = polygons_to_mask(top_contour, height, width)

    # sharpe_kernel =  np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # sharpen_image = cv.filter2D(tree_area, cv.CV_32F, kernel=sharpe_kernel)
    # sharpen_image = cv.convertScaleAbs(sharpen_image)

    # plt.imshow(sharpen_image, cmap='gray')
    # plt.show()

    # Show in a window
    is_include = check_contour_include(top_contour[0], top_contour[1])

    print(is_include)

    cv.drawContours(image, top_contour, -1, (0, 0, 255), 3)
    plt.axis('off')
    plt.imshow(image[:, :, ::-1])
    plt.show()



def get_leaf_area(image, visualize=False):


    hsv_img = cv.cvtColor(image, code=cv.COLOR_RGB2HSV)
    low_threshold = (45, 45, 5)
    upper_threshold = (255, 255, 255)

    h_channel = hsv_img[:, :, 0]

    # adaptive_binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)

    # use h channels
    h_mask = (h_channel > 40) & (h_channel < 90)

    # use unit hsv space
    hsv_mask = cv.inRange(hsv_img, lowerb=low_threshold, upperb=upper_threshold) / 255

    tree_mask = np.asarray(h_mask * hsv_mask, dtype=np.uint8)

    # morphology process
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # tree_mask = cv.morphologyEx(tree_mask, cv.MORPH_OPEN, kernel)
    # tree_mask = cv.morphologyEx(tree_mask, cv.MORPH_CLOSE, kernel, iterations=3)
    # tree_mask = cv.dilate(tree_mask, kernel=kernel, iterations=3)


    if visualize:

        tree_area = cv.bitwise_and(image, image, mask=tree_mask)
        plt.imshow(tree_area[:, :, ::-1])
        plt.axis('off')
        plt.show()

    return tree_mask


def occlusion_detect(image, threshold=0.2):

    h, w, _ = image.shape

    tree_mask = get_leaf_area(image, visualize=True)

    tree_rate = np.sum(tree_mask) / (h*w)

    return tree_rate > threshold



def main():


    bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)

    print(occlusion_detect(bgr_img))

    print('Done !')


if __name__ == "__main__":
    main()

