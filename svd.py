#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : svd.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/12/22 上午8:47
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


np.random.seed(2020)


def svd_demo():
    a_0 = np.random.randint(1, 5, size=12).reshape((3, 4))
    print(a_0)

    a_a_t = np.matmul(a_0, a_0.T)

    a_t_a = np.matmul(a_0.T, a_0)

    u_0, sigma_0, v_0 = np.linalg.svd(a_a_t)  # equal to evd
    u_1, sigma_1, v_1 = np.linalg.svd(a_t_a)  # equal to evd

    u, sigma, v = np.linalg.svd(a_0)

    print(u.shape, sigma.shape, v.shape)

    assert u_0.all() == v_0.all() == u.all()
    assert u_1.all() == v_1.all() == v.all()


def image_compress(image, num_singular=120):

    h, w, c = image.shape
    print('image shape')

    # reshape
    reshape_img = image.reshape(h, w * c)
    # svd
    u, sigma, v = np.linalg.svd(reshape_img)
    print(u.shape, sigma.shape, v.shape)

    compress_img = u[:, :num_singular].dot(np.diag(sigma[:num_singular])).dot(v[:num_singular, :])
    compress_img = compress_img.reshape(h, w, c)
    compress_img = compress_img.astype(np.uint8)

    plt.imshow(compress_img)
    plt.axis('off')
    plt.show()

    return compress_img


image_path = './images/demo.jpg'

def main():
    # svd demo
    svd_demo()

    # image compress
    # read
    rgb_img = plt.imread(image_path)

    img_compress = image_compress(rgb_img)
    print('Done')


if __name__ == "__main__":
    main()
