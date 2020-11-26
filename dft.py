#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dft.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/25 下午2:38
# @ Software   : PyCharm
#-------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2 as cv

from tools import visual_fft_magnitude, visual_fft_phase, visual_magnitude_with_opencv


barbara_path = './images/DIP3E_CH4/Fig0417(a)(barbara).tif'
rectangle_path = './images/DIP3E_CH4/Fig0424(a)(rectangle).tif'
translated_rectangle_path = './images/DIP3E_CH4/Fig0425(a)(translated_rectangle).tif'

def visual_fft_spectrum_demo(spectrum, title='fft_spectrum'):


    # load real part
    real_f = np.abs(spectrum)
    # convert scale to (0, 255)
    min = np.min(real_f)
    max = np.max(real_f)
    scale_f = -min + 255 / (max - min) * real_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()


def visual_fft():
    gray_img = plt.imread(rectangle_path)
    # ---------------------------------- origin fft spectrum--------------------------------
    f = np.fft.fft2(gray_img)
    #
    # visual_fft_spectrum_demo(f, 'origin fft spectrum')
    # #----------------------------------- centralize spectrum--------------------------------
    # central_f = np.fft.fftshift(f)
    # visual_fft_spectrum_demo(central_f, 'central fft spectrum')
    # #----------------------------------- change spectrum amplitude--------------------------
    # scale_f = np.log(1 + np.abs(central_f))
    # visual_fft_spectrum_demo(scale_f, 'scale fft spectrum')
    visual_fft_magnitude(f, 'origin fft spectrum')
    visual_fft_phase(f, 'origin fft phase')

    # ----------------------------------- translate-----------------------------------------
    translated_img = plt.imread(translated_rectangle_path)
    plt.imshow(translated_img, cmap='gray')
    translated_f = np.fft.fft2(translated_img)
    visual_fft_magnitude(translated_f, 'translated fft spectrum')
    visual_fft_phase(translated_f, 'translated fft phase')
    #----------------------------------- image rotation--------------------------------------
    rotation_img = ndimage.rotate(gray_img, angle=45)
    plt.imshow(rotation_img, cmap='gray')
    rotation_f = np.fft.fft2(rotation_img)
    visual_fft_magnitude(rotation_f, 'rotation fft spectrum')
    visual_fft_phase(rotation_f, 'translated fft phase')
    plt.show()
    print('Done')


def blur_with_numpy(image, size=30):
    """
    low_pass
    :param image:
    :param size:
    :return:
    """
    assert len(image.shape) == 2

    img_h, img_w = image.shape
    # padding with zero
    # padding_img = np.zeros((3*img_h, 3*img_w), dtype=np.uint8)
    # padding_img[0:img_h, 0:img_w] = image

    # construct filter mask
    center_y, center_x = int(img_h / 2), int(img_w / 2)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[center_y - size:center_y + size, center_x - size:center_x + size] = 1

    # execute fft transform
    f = np.fft.fft2(image)
    # centralize
    shift_f = np.fft.fftshift(f)
    # frequency filter
    shift_f *= mask

    # invert centralize
    ishift_f = np.fft.ifftshift(shift_f)
    ifft_img = np.fft.ifft2(ishift_f)

    #
    blur_img = np.abs(ifft_img)

    return blur_img, mask


def sharp_with_high_pass(image, size=30, alpha=0.85):
    """

    :return:
    """
    assert len(image.shape) == 2

    img_h, img_w = image.shape
    # padding with zero
    # padding_img = np.zeros((3*img_h, 3*img_w), dtype=np.uint8)
    # padding_img[0:img_h, 0:img_w] = image

    # construct filter mask
    center_y, center_x = int(img_h / 2), int(img_w / 2)
    mask = np.ones_like(image, dtype=np.float32)
    mask[center_y - size:center_y + size, center_x - size:center_x + size] = alpha

    # execute fft transform
    f = np.fft.fft2(image)
    # centralize
    shift_f = np.fft.fftshift(f)
    # frequency filter
    shift_f *= mask

    # invert centralize
    ishift_f = np.fft.ifftshift(shift_f)
    ifft_img = np.fft.ifft2(ishift_f)

    filter_img = np.abs(ifft_img)

    sharp_img = np.clip(filter_img, 0, 255).astype(np.uint8)

    return sharp_img, mask


def blur_with_opencv(image, size=30):

    assert len(image.shape) == 2

    # -----------------Expand the image to an optimal size with padding-------------------
    rows, cols = image.shape
    m = cv.getOptimalDFTSize(rows)
    n = cv.getOptimalDFTSize(cols)
    padded = cv.copyMakeBorder(image, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # -------------------------Make place for both the complex and the real values------------------
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complex_img = cv.merge(planes)  # Add to the expanded another plane with zeros

    # ---------------------construct mask----------------------------
    center_y, center_x = int(complex_img.shape[0] / 2), int(complex_img.shape[1] / 2)
    mask = np.zeros_like(complex_img, dtype=np.uint8)
    mask[center_y - size:center_y + size, center_x - size:center_x + size, :] = 1

    # -------------------------Make the Discrete Fourier Transform-----------------------------------
    dft_img = cv.dft(complex_img)  # this way the result may fit in the source matrix

    # -------------------------ransform the real and complex values to magnitude--------------------
    # compute the magnitude and switch to logarithmic scale
    # = > M = sqrt(Re(DFT(I)) ^ 2 + Im(DFT(I)) ^ 2)
    # cv.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    # cv.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    # image_ = planes[0]

    shift_img = np.fft.fftshift(dft_img)
    shift_img *= mask

    # invert centralize
    ishift_img = np.fft.ifftshift(shift_img)
    ifft_img = cv.idft(ishift_img)

    blur_img, img_1 = cv.split(ifft_img)

    # convert scale to (0, 1)
    cv.normalize(blur_img, blur_img, 0, 1, cv.NORM_MINMAX)

    # recover origin shape
    blur_img = blur_img[0:rows, 0:cols]

    return blur_img

# def main():
#     gray_imag = cv.imread(barbara_path, cv.IMREAD_GRAYSCALE)
#     sharp_img, high_mask = sharp_with_high_pass(gray_imag)
#     plt.imshow(sharp_img, cmap='gray')
#     plt.show()

def main():

    image = cv.imread(barbara_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print('Error opening image')
        return -1

    # mag_img = visual_magnitude_with_opencv(image)
    # [normalize]
    blur_img = blur_with_opencv(image)
    cv.imshow("Input Image"       , image)    # Show the result
    cv.imshow("spectrum magnitude", blur_img)
    cv.waitKey()

if __name__ == "__main__":
    main()
    # print(100 & -2)