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

import numpy as np
import cv2 as cv


class GeneratorNoise(object):
    """
    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
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

def main():
    img_path = './images/cat.jpg'
    image = cv.imread(img_path, flags=cv.IMREAD_COLOR)

    generator_noise = GeneratorNoise()
    gauss_img = generator_noise.gauss_noise(image)
    sp_img = generator_noise.salt_pepper_noise(image, rate=0.1)
    cv.imshow('gauss image', gauss_img)
    cv.imshow('s&p image', sp_img)
    cv.waitKey(0)



if __name__ == "__main__":
    main()