#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/24 上午11:32
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt


def show_histogram(image, title='hist'):
    """

    :param image:
    :param channel:
    :param title:
    :return:
    """

    # fig, axs = plt.subplots(2, 2)
    # # draw histogram
    # plt.hist(image.ravel(), bins=20)
    assert len(image.shape) in [2, 3]
    if image.shape == 3:
        # display split channel
        img_channel = cv.split(image)
        colors = ['b', 'g', 'r']
        for color, channel in zip(colors, img_channel):
            sns.distplot(channel.ravel(), bins=25, kde=True, label=color)
        plt.title('RGB histogram')
        plt.legend()
        plt.show()

        # gray histogram
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sns.distplot(gray_image.ravel(), bins=25, kde=True)
        plt.title('GRY histogram')
        plt.show()
    else:
        sns.distplot(image.ravel(), bins=25, kde=True)
        plt.title('GRY histogram')
        plt.show()


def visual_fft_magnitude(fft, title='fft_magnitude'):
    """

    :param spectrum:
    :param title:
    :return:
    """
    # centralize
    central_f = np.fft.fftshift(fft)
    # real value
    abs_f = np.abs(central_f)
    # change scale
    scale_f = np.log(1 + abs_f)

    # convert scale to (0, 255)
    min = np.min(scale_f)
    max = np.max(scale_f)
    scale_f = -min + 255 / (max - min) * scale_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()


def visual_fft_phase(fft, title='fft_phase'):
    """

    :param fft:
    :param title:
    :return:
    """
    # centralize
    central_f = np.fft.fftshift(fft)
    real_f = np.real(central_f)
    imag_f = np.imag(central_f)
    phase_f = np.arctan2(imag_f, real_f)
    # convert scale to (0, 255)
    min = np.min(phase_f)
    max = np.max(phase_f)
    scale_f = -min + 255 / (max - min) * phase_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()


def visual_magnitude_with_opencv(image):
    """"""
    assert len(image.shape) == 2
    # -----------------Expand the image to an optimal size with padding-------------------
    rows, cols = image.shape
    m = cv.getOptimalDFTSize(rows)
    n = cv.getOptimalDFTSize(cols)
    padded = cv.copyMakeBorder(image, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # -------------------------Make place for both the complex and the real values------------------
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complex_img = cv.merge(planes)  # Add to the expanded another plane with zeros

    # -------------------------Make the Discrete Fourier Transform-----------------------------------
    dft_img = cv.dft(complex_img)  # this way the result may fit in the source matrix

    # ---------------------compute the magnitude and switch to logarithmic scale-----------------------
    # = > log(1 + sqrt(Re(DFT(I)) ^ 2 + Im(DFT(I)) ^ 2))
    ## [magnitude]
    cv.split(dft_img, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    mag_img = cv.magnitude(planes[0], planes[1])  # planes[0] = magnitude

    # ------------------------Switch to a logarithmic scale-----------------------------------------
    # = > log(1 + M)
    mat_ones = np.ones(mag_img.shape, dtype=mag_img.dtype)
    cv.add(mat_ones, mag_img, mag_img)  # switch to logarithmic scale
    cv.log(mag_img, mag_img)  # log

    # ------------------------Centralize----------------------------------------------------

    # magI_rows, magI_cols = magI.shape
    # # crop the spectrum, if it has an odd number of rows or columns
    # magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]  # convert to even
    # cx = int(magI_rows/2)
    # cy = int(magI_cols/2)
    #
    # q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    # q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    # q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    # q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right
    #
    # tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    # magI[0:cx, 0:cy] = q3
    # magI[cx:cx + cx, cy:cy + cy] = tmp
    #
    # tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    # magI[cx:cx + cx, 0:cy] = q2
    # magI[0:cx, cy:cy + cy] = tmp
    mag_img = np.fft.fftshift(mag_img)

    # --------------------------------Normalize--------------------------------------------
    # Transform the matrix with float values into a viewable image form(float between values 0 and 1).
    cv.normalize(mag_img, mag_img, 0, 1, cv.NORM_MINMAX)

    return mag_img