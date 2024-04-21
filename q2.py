# -*- coding: utf-8 -*-
"""
Template for homework exercise 5, question 2
Fundamentals of CS for EE students 2023
"""


import numpy as np
from skimage import data
from skimage.color import rgb2gray
from matplotlib import pyplot as plt



def gaussian_kernel(row, col, mean_v=0, std_v=None, mean_h=0, std_h=None):

    e = np.e
    y_pow_lambda = lambda y: (np.power(y - mean_v, 2)) / (2 * np.power(std_v, 2))
    x_pow_lambda = lambda x: (np.power(x - mean_h, 2)) / (2 * np.power(std_h, 2))
    up_frac = lambda x, y: np.power(e, -1 * ((x_pow_lambda(x) + y_pow_lambda(y))))
    gaussian = lambda x, y: up_frac(x, y) / (2 * np.pi * std_v * std_h)  # final function of Normal
    x_axis = np.arange(-col/2,col/2)
    y_axis = np.arange(-row/2, row/2)

    # catch exption if needed
    if std_v is None:
        std_v = y_axis / 2
    if std_h is None:
        std_h = x_axis / 2
    if std_h == 0:
        raise Exception
    if std_v == 0:
        raise Exception

    x, y = np.meshgrid(x_axis, y_axis)  # output two type, x and y

    return gaussian(x, y)


def gaussian_blur(image, g_ker):
    """Applies gaussian blurring to input image using Gaussian kernel g_ker"""
    kernal_row_col = g_ker.shape  # kernal row and col
    kernal_row = kernal_row_col[0]
    kernal_col = kernal_row_col[1]

    image_row_col = image.shape  # image row and col
    image_row = (image_row_col[0])
    image_col = (image_row_col[1])
    matrix = np.ones(image_row_col)  # new image
    pad_image = np.pad(image, ((0, int(kernal_row)), (0,int(kernal_col))), mode="constant", constant_values=0)  # pad only the bottom and right side
    for row in range(image_row):
        for col in range(image_col):
            matrix[row, col] = np.sum(pad_image[row:row + kernal_row, col:col + kernal_col] * g_ker)
    matrix = image.kernel_version()

    return matrix












if __name__ == '__main__':
    image = data.astronaut()
    image_grey = rgb2gray(image)  # converts color images to black-and-white
    plt.imshow(image)
    plt.imshow(image_grey, cmap='gray')
    plt.imsave('q2a.png', image_grey)

    g_ker = gaussian_kernel(100, 50, mean_v=0, std_v=15, mean_h=-0, std_h=10)
    # plt.imshow(g_ker, cmap='gray')
    # plt.imsave('q2b.png', g_ker)
    #

    image_blur = gaussian_blur(image_grey, g_ker)
    # plt.imshow(image_blur, cmap='gray')
    # plt.imsave('q2c.png', image_blur)


g = gaussian_kernel(50, 100,  mean_v=-10, std_v=10, mean_h=0, std_h=10)
# plt.imshow(g, cmap='gray')
# plt.savefig('q2b.png')
# plt.show()
