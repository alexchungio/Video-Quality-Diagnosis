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
from skimage.measure import label

img_path = './images/occlusion/occlusion_3.jpg'
# img_path = './images/demo_1.jpg'


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



def get_leaf_area(hsv_img, visualize=False):

    """

    :param image:
    :param visualize:
    :return:
    """
    assert len(hsv_img.shape) == 3
    # range of green in H channel
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
        image = cv.cvtColor(hsv_img, code=cv.COLOR_HSV2BGR)
        tree_area = cv.bitwise_and(image, image, mask=tree_mask)
        plt.imshow(tree_area[:, :, ::-1])
        plt.axis('off')
        plt.show()

    return tree_mask


def occlusion_detect_with_leaf(hsv_img, threshold=0.2, visualize=False):

    assert len(hsv_img.shape) == 3
    h, w, _ = hsv_img.shape

    tree_mask = get_leaf_area(hsv_img, visualize=visualize)

    if visualize:
        plt.imshow(tree_mask, cmap='gray')
        plt.show()

    tree_rate = np.sum(tree_mask) / (h * w)

    return tree_rate > threshold


def get_largest_connect(bw_img):
    '''
    compute largest Connect component of a binary image

    Parameters:
    ---

    bw_img: ndarray
        binary image

	Returns:
	---

	lcc: ndarray
		largest connect component.


    '''
    labeled_img, num = label(bw_img, neighbors=4, background=1, return_num=True)

    max_label = 0
    max_num = 0
    # compute max connect
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc, max_num


def occlusion_detect_with_gray(gray_img, threshold=0.2, visulize=False):
    """

    :param image:
    :param threshold:
    :return:
    """
    assert len(gray_img.shape) == 2
    h, w = gray_img.shape
    # img = cv.adaptiveThreshold(gray_img, 255 ,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    # make low region as foreground
    # _, otsu_binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    _, binary_img = cv.threshold(blur, 50, 1, cv.THRESH_BINARY)

    lcc, sum_lcc = get_largest_connect(binary_img)

    if visulize:
        plt.imshow(lcc, cmap='gray')
        plt.show()

    low_rate = sum_lcc / (h * w)

    return low_rate > threshold


def occlusion_detect(gray_img, hsv_img, threshold_0=0.2, threshold_1=0.25, visualize=False):

    gray_occlusion = occlusion_detect_with_gray(gray_img, threshold_0, visualize)

    leaf_occlusion = occlusion_detect_with_leaf(hsv_img, threshold_1, visualize)

    return leaf_occlusion or gray_occlusion



def main():


    bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    bgr_img = cv.resize(bgr_img, (128, 128))

    gray_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2GRAY)
    hsv_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2HSV)
    # print(occlusion_detect(bgr_img))
    # is_occlusion_0 = occlusion_detect_with_leaf(bgr_img, visualize=True)
    is_occlusion = occlusion_detect(hsv_img=hsv_img, gray_img=gray_img)
    print(is_occlusion)
    print('Done !')


if __name__ == "__main__":
    main()

