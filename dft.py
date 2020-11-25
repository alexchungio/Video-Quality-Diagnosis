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
import matplotlib.transforms as tr
import numpy as np
from scipy import ndimage

from tools import visual_fft_spectrum, visual_fft_phase


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


def main():

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
    visual_fft_spectrum(f, 'origin fft spectrum')
    visual_fft_phase(f, 'origin fft phase')

    # ----------------------------------- translate-----------------------------------------
    translated_img = plt.imread(translated_rectangle_path)
    plt.imshow(translated_img, cmap='gray')
    translated_f = np.fft.fft2(translated_img)
    visual_fft_spectrum(translated_f, 'translated fft spectrum')
    visual_fft_phase(translated_f, 'translated fft phase')
    #----------------------------------- image rotation--------------------------------------
    rotation_img = ndimage.rotate(gray_img, angle=45)
    plt.imshow(rotation_img, cmap='gray')
    rotation_f = np.fft.fft2(rotation_img)
    visual_fft_spectrum(rotation_f, 'rotation fft spectrum')
    visual_fft_phase(rotation_f, 'translated fft phase')
    plt.show()
    print('Done')


if __name__ == "__main__":
    main()
