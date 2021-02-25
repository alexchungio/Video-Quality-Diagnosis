#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : snow_noise_detect.py
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

    def snow_noise(self, image, rate=0.05):
        """

        :param image:
        :param rate:
        :return:
        """
        noise_img = np.zeros_like(image, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                randn = np.random.random()
                # salt mode
                if randn < rate:
                    noise_img[i, j, :] = 255
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
        gauss = gauss.reshape(row,col, ch)
        noisy = image + image * gauss
        return noisy



def snow_noise_detect(image, center_rate=0.1, threshold=0.5, visual=False):
    """
    detect high frequency percent
    :param image:
    :param center_rate:
    :param threshold:
    :return:
    """
    assert len(image.shape) == 2
    (h, w) = image.shape
    center_x, center_y = int(w / 2.0), int(h / 2.0)
    # gray_img = cv.imread(image, flags=cv.IMREAD_GRAYSCALE)

    fft = np.fft.fft2(image)
    # centralize
    fftShift = np.fft.fftshift(fft)

    # real value
    # abs_f = np.abs(central_f)
    # # change scale
    # scale_f = np.log(1 + abs_f)

    row_length = int(h * center_rate)
    col_length = int(w * center_rate)


    mask = np.zeros_like(image, dtype=np.float32)
    mask[center_y - row_length:center_y + row_length, center_x - col_length:center_x + col_length] = 1

    # central_f[center_y - size:center_y + size, center_x - size:center_x + size] = 0
    fftShift = fftShift * mask

    ifftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(ifftShift)

    # center_percent = np.sum(fft_center) / np.sum(scale_f)
    if visual:
        # compute the magnitude spectrum of the transform
        mag_spectrum = 20 * np.log(np.abs(fftShift))

        filter_img = (np.abs(recon))
        # display the original input image
        (fig, ax) = plt.subplots(1, 3, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(mag_spectrum, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(filter_img, cmap="gray")
        ax[2].set_title("Filter Image")
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        # show our plots
        plt.show()


    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    filter_img = np.abs(recon).astype(np.uint8)

    plt.imshow(filter_img, cmap='gray')
    plt.show()

    mean = np.mean(image - filter_img) / 255

    print(mean)

    return mean > threshold


def main():
    img_path = './images/demo.jpg'
    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    generator_noise = GeneratorNoise()

    snow_img = generator_noise.snow_noise(image, rate=0.2)
    snow_gray_img = cv.cvtColor(snow_img, cv.COLOR_BGR2GRAY)

    # visual_fft_magnitude(gray_img)
    # visual_fft_magnitude(gray_sp)
    # cv.imshow('gauss image', gauss_img)
    # cv.imshow('s&p image', gray_sp)
    # cv.waitKey(0)
    # center_rate = 0.005
    # print(noise_detect(gray_sp, center_rate=center_rate) / noise_detect(gray_img, center_rate=center_rate))

    print(snow_noise_detect(gray_img, visual=True))
    print(snow_noise_detect(snow_gray_img, visual=True))


if __name__ == "__main__":
    main()