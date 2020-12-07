#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : noise_detect.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/27 下午2:00
# @ Software   : PyCharm
#-------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tools import visual_fft_magnitude


class GeneratorNoise(object):
    """
    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'stripe'   Multiplicative noise using out = image + n*image,where
                n,is uniform noise with specified mean & variance.
    """
    def __init__(self):
        pass

    def gauss_noise(self, image, mean=0.0, var=0.001):

        image = image.astype(np.float32) / 255.
        gauss_noise = np.random.normal(mean, var ** 0.5, size=image.shape)

        noise_img = image + gauss_noise
        noise_img = np.clip(noise_img, 0, 1)
        noise_img = (noise_img * 255).astype(np.uint8)

        return noise_img

    def salt_pepper_noise(self, image, rate=0.05):

        noise_img = np.zeros_like(image, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                randn = np.random.random()
                # salt mode
                if randn < rate:
                    noise_img[i, j, :] = 255
                # pepper mode
                elif randn > 1 - rate:
                    noise_img[i, j, :] = 0
                else:
                    noise_img[i, j, :] = image[i, j, :]

        return noise_img

    def poisson_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    def speckle_noise(self, image):
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy



def noise_detect(image, center_rate=0.005, threshold=0.01, visual=False):
    """
    detect high frequency percent
    :param image:
    :param center_rate:
    :param threshold:
    :return:
    """
    assert len(image.shape) == 2
    rows, cols = image.shape
    # gray_img = cv.imread(image, flags=cv.IMREAD_GRAYSCALE)

    fft = np.fft.fft2(image)
    # centralize
    central_f = np.fft.fftshift(fft)
    # real value
    abs_f = np.abs(central_f)
    # change scale
    scale_f = np.log(1 + abs_f)

    row_length = int(rows * center_rate)
    col_length = int(cols * center_rate)

    row_low = int((rows - row_length) / 2)
    row_high = row_low + row_length

    col_low = int((cols - col_length) / 2)
    col_high = col_low + col_length

    mask = np.zeros_like(image, dtype=np.float32)

    mask[row_low:row_high, :] = 1
    mask[:, col_low:col_high] = 1

    fft_center = scale_f * mask

    center_percent = np.sum(fft_center) / np.sum(scale_f)
    if visual:
        cv.normalize(fft_center, fft_center, 0, 1, cv.NORM_MINMAX)
        fft_center *= 255.
        fft_center = fft_center.astype(np.uint8)

        plt.imshow(fft_center, cmap='gray')
        plt.show()

    return center_percent < threshold


def main():
    img_path = './images/blur/blur_1.jpg'
    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    generator_noise = GeneratorNoise()
    gauss_img = generator_noise.gauss_noise(image)
    sp_img = generator_noise.salt_pepper_noise(image, rate=0.1)
    gray_sp = cv.cvtColor(sp_img, cv.COLOR_BGR2GRAY)
    # visual_fft_magnitude(gray_img)
    # visual_fft_magnitude(gray_sp)
    # cv.imshow('gauss image', gauss_img)
    # cv.imshow('s&p image', gray_sp)
    # cv.waitKey(0)
    # center_rate = 0.005
    # print(noise_detect(gray_sp, center_rate=center_rate) / noise_detect(gray_img, center_rate=center_rate))

    print(noise_detect(gray_img))
    print(noise_detect(gray_sp))


if __name__ == "__main__":
    main()