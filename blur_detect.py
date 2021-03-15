#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : blur_detect.py
# @ Description:  https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/24 下午3:23
# @ Software   : PyCharm
#-------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from tools import visual_fft_magnitude


def detect_blur_fft(image, size=60, threshold=20, vis=False):
    """

    :param image:
    :param size:
    :param threshold:
    :param vis:
    :return:
    """
    assert len(image.shape) == 2, "Image format must be gray"
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (center_x, center_y) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[center_y - size:center_y + size, center_x - size:center_x + size] = 0

    ifftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(ifftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    # plt.imshow(np.abs(recon))
    # plt.show()
    magnitude = 20 * np.log(np.abs(recon))

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        mag_spectrum= 20 * np.log(np.abs(fftShift))

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

    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= threshold)


def test(gray_img, size=60, threshold=20, visual=False):
    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray_img, size=size,
                                     threshold=threshold, vis=visual)

    # draw on the image, indicating whether or not it is blurry
    image = np.dstack([gray_img] * 3)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv.putText(image, text, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               color, 2)
    print("[INFO] {}".format(text))
    # show the output image
    # cv.imshow("Output", image)
    # cv.waitKey(0)
    #
    # cv.destroyAllWindows()

def eval(gray_img, size=60, threshold=20, visual=False):

    # check to see if are going to test our FFT blurriness detector using
    # various sizes of a Gaussian kernel
    # loop over various blur radii
    for radius in range(1, 30, 2):
        # clone the original grayscale image
        image = gray_img.copy()

        # check to see if the kernel radius is greater than zero
        if radius > 0:
            # blur the input image by the supplied radius using a
            # Gaussian kernel
            image = cv.GaussianBlur(image, (radius, radius), 0)

            # apply our blur detector using the FFT
            (mean, blurry) = detect_blur_fft(image, size=size,
                                             thresh=threshold, vis=visual)

            # draw on the image, indicating whether or not it is
            # blurry
            image = np.dstack([image] * 3)
            color = (0, 0, 255) if blurry else (0, 255, 0)
            text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
            text = text.format(mean)
            cv.putText(image, text, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            print("[INFO] Kernel: {}, Result: {}".format(radius, text))

        # show the image
        cv.imshow("Test Image", image)
        cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img_path = './images/blur/blur_2.jpg'
    size = 60 # low frequencies size
    threshold = 20 # threshold for our blur detector to fire
    visual = True #b whether or not we are visualizing intermediary steps

    bgr_img = cv.imread(img_path, cv.IMREAD_COLOR)
    gray_image = cv.cvtColor(bgr_img, code=cv.COLOR_BGR2GRAY)
    # f = np.fft.fft2(gray_image)
    # visual_fft_spectrum(f)

    visual_fft_magnitude(gray_image)
    test(gray_image, size, threshold, visual)

    print('Done')

if __name__ == "__main__":
    main()
