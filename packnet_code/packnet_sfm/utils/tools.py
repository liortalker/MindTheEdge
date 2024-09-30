# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

import cv2
import numpy as np
import os
import scipy.ndimage

def non_max_suppression(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_angle = np.arctan2(sobely,sobelx)

    H, W = img.shape
    out = np.zeros((H, W))
    angle = np.rad2deg(sobel_angle)
    angle[angle < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 1
            r = 1
            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i + 1, j]
                r = img[i - 1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                out[i, j] = img[i, j]
            else:
                out[i, j] = 0


    return out

# Hysteresis Thresholding
def hysteresis(img, t_low=0.3, t_high=0.7):

    temp_img = np.copy(img)

    # Assign values to pixels
    for i in range(1, int(img.shape[0] - 1)):
        for j in range(1, int(img.shape[1] - 1)):
            # Strong pixels
            if (img[i, j] > t_high):
                temp_img[i, j] = 2
            # Weak pixels
            elif (img[i, j] < t_low):
                temp_img[i, j] = 0
            # Intermediate pixels
            else:
                temp_img[i, j] = 1

    # Include weak pixels that are connected to chain of strong pixels
    total_strong = np.sum(temp_img == 2)
    while (1):
        DFS(temp_img)
        if (total_strong == np.sum(temp_img == 2)):
            break
        total_strong = np.sum(temp_img == 2)

    # Remove weak pixels
    for i in range(1, int(temp_img.shape[0] - 1)):
        for j in range(1, int(temp_img.shape[1] - 1)):
            if (temp_img[i, j] == 1):
                temp_img[i, j] = 0

    temp_img = temp_img / np.max(temp_img)

    return img*temp_img

#Function to include weak pixels that are connected to chain of strong pixels
def DFS(img) :
    for i in range(1, int(img.shape[0] - 1)) :
        for j in range(1, int(img.shape[1] - 1)) :
            if(img[i, j] == 1) :
                t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                            img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                if(t_max == 2) :
                    img[i, j] = 2

def remove_isolated_edges(img):

    tmp = scipy.ndimage.convolve(img, np.ones((3, 3)), mode='constant')
    img = np.logical_and(tmp >= 2, img).astype(np.float32)

    return img


def LIDARsample(depth, f=960, u0=960, v0=540, lidar_pitch=0, lidar_beams=64, lidar_ver_angle=26.8, lidar_hor_res=0.09, max_depth=120, FOV=90):

    W = depth.shape[1]
    H = depth.shape[0]

    gridx, gridy = np.meshgrid(np.arange(W),np.arange(H))

    x = (gridx-u0)/f * depth
    y = (v0-gridy)/f * depth

    anglex = np.arctan2(x,np.sqrt(depth**2+y**2))
    angley = np.arctan2(y,np.sqrt(depth**2+x**2))

    hor_samp = np.linspace(np.deg2rad(-FOV/2),np.deg2rad(FOV/2),int(FOV/lidar_hor_res))
    ver_sample = np.linspace(np.min(angley)+lidar_pitch,np.min(angley) + np.deg2rad(lidar_ver_angle),lidar_beams)

    ver_sample_reshaped = ver_sample.reshape(1,lidar_beams)
    hor_samp_reshaped = hor_samp.reshape(1, int(FOV / lidar_hor_res))

    rows_sample = np.zeros_like(depth)
    for i in np.arange(W):
        angley_col = angley[:,i].reshape(H,1)
        inds = np.argmin(np.abs(angley_col-ver_sample_reshaped),axis=0)
        rows_sample[inds,i] = 1+np.arange(lidar_beams)

    lidar_mask = np.zeros_like(depth)
    for i in np.arange(1,1+lidar_beams):
        indsi = np.where(rows_sample==i)
        anglex_row = anglex[rows_sample == i].reshape(-1, 1)
        inds = np.argmin(np.abs(anglex_row - hor_samp_reshaped), axis=0)
        eee = np.array(indsi)[:, inds]
        lidar_mask[eee[0], eee[1]] = 1

    lidar_mask[depth>max_depth]=0

    lidar = np.zeros_like(lidar_mask)
    lidar[lidar_mask==1] = depth[lidar_mask==1]

    return lidar

def checkIfExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
