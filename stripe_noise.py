#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : stripe_noise.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/7 上午10:08
# @ Software   : PyCharm
#-------------------------------------------------------



import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from tools import visual_fft_magnitude


img_path = './images/stripe/stripe_1.jpg'
# img_path = './images/demo.jpg'


def detect_stripe_with_fft(image, size=30, threshold=10, visualize=False):
    """

    :param image:
    :param size:
    :param threshold:
    :param visualize:
    :return
    """

    assert len(image.shape) == 2, "Image format must be gray"

    h, w = image.shape
    (center_x, center_y) = (int(w / 2.0), int(h / 2.0))

    # fft
    fft = np.fft.fft2(image)
    # centralize
    central_fft = np.fft.fftshift(fft)
    # real value
    abs_fft = np.abs(central_fft)

    # scale transform
    cv.normalize(abs_fft, abs_fft, 0, 1, cv.NORM_MINMAX)
    # suppress the direct component value of fft
    abs_fft *= 500
    abs_fft = np.clip(abs_fft, 0, 255)
    abs_fft = abs_fft.astype(np.uint8)

    # mask operation
    mask_center_fft = abs_fft.copy()
    mask_center_fft[center_y - size:center_y + size, center_x - size:center_x + size] = 0

    if visualize:
        (fig, ax) = plt.subplots(1, 3, )
        for i, (img, name) in enumerate(zip([image, abs_fft, mask_center_fft], ['input', 'center', 'center & mask'])):
            ax[i].imshow(img, cmap="gray")
            ax[i].set_title(name)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()

    # get number of highlight point
    grad_fft = cv.Laplacian(mask_center_fft, cv.CV_32F)
    highlight_point = grad_fft > 4
    num_highlight = np.sum(highlight_point)

    print(num_highlight)

    return num_highlight > threshold



def main():

    bgr_img = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    # gray_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2GRAY)
    gray_img = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2HSV)

    stripe_flag = detect_stripe_with_fft(gray_img[:, :, 0], visualize=True)
    print(stripe_flag)

    print('Done')


if __name__ == "__main__":
    main()