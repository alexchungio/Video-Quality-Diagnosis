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
import cv2 as cv


barbara_path = './images/DIP3E_CH4/Fig0417(a)(barbara).tif'
rectangle_path = './images/DIP3E_CH4/Fig0424(a)(rectangle).tif'
rectangle_translated_path = './images/DIP3E_CH4/Fig0424(a)(rectangle).tif'

def visual_fft_spectrum(spectrum, title='fft_spectrum'):
    # load real part
    real_f = np.real(spectrum)
    # convert scale to (0, 255)
    min = np.min(real_f)
    max = np.max(real_f)
    scale_f = -min + 255 / (max - min) * real_f
    scale_f = scale_f.astype(np.uint8)
    plt.title(title)
    plt.imshow(scale_f, cmap='gray')
    plt.show()

def main():

    gray_img = plt.imread(barbara_path)

    # ---------------------------------- origin fft spectrum--------------------------------
    f = np.fft.fft2(gray_img)

    visual_fft_spectrum(f, 'origin fft spectrum')
    #----------------------------------- centralize spectrum--------------------------------
    central_f = np.fft.fftshift(f)
    visual_fft_spectrum(central_f, 'central fft spectrum')
    #----------------------------------- change spectrum amplitude--------------------------
    scale_f = np.log(1 + np.abs(central_f))
    visual_fft_spectrum(scale_f, 'scale fft spectrum')
    print('Done')

if __name__ == "__main__":
    main()