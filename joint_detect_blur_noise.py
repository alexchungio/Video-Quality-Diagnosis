#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : joint_detect_blur_noise.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/24 下午2:12
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from snow_noise_detect import GeneratorNoise


def detect_blur_noise(gray_img, size=60, blur_threshold=20, noise_threshold=50, visulize=False):
    """

    :param image:
    :param size:
    :param thresh:
    :param vis:
    :return:
    """
    assert len(gray_img.shape) == 2, "Image format must be gray"
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = gray_img.shape
    (center_x, center_y) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
    fft = np.fft.fft2(gray_img)
    fftShift = np.fft.fftshift(fft)


    # check to see if we are visualizing our output
    if visulize:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(gray_img, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[center_y - size:center_y + size, center_x - size:center_x + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value

    blur_flag = mean < blur_threshold
    noise_flag = mean > noise_threshold

    return blur_flag, noise_flag


def main():
    generator_noise = GeneratorNoise()


    img_path = './images/demo.jpg'
    blur_img_path = './images/blur/blur_0.jpg'


    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sp_img = generator_noise.salt_pepper_noise(image, rate=0.1)
    gray_sp = cv.cvtColor(sp_img, cv.COLOR_BGR2GRAY)
    print(detect_blur_noise(gray_img))
    print(detect_blur_noise(gray_sp))

    image = cv.imread(blur_img_path, flags=cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(detect_blur_noise(gray_img))
    sp_img = generator_noise.salt_pepper_noise(image, rate=0.1)
    gray_sp = cv.cvtColor(sp_img, cv.COLOR_BGR2GRAY)
    print(detect_blur_noise(gray_sp))




if __name__ == "__main__":
    main()