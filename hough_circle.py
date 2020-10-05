from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def main():
    MYPATH = os.getcwd()
    fnames_kmeans = os.listdir(MYPATH + '/img_kmeans')
    fnames_norm = os.listdir(MYPATH + '/img_norm')
    fnames_orig = os.listdir(MYPATH + '/img_orig')

    folders = ['kmeans', 'norm', 'orig']
    fnames = [fnames_kmeans, fnames_norm, fnames_orig]

    print(f'fnames_norm: {fnames_norm}\n fnames_kmeans: {fnames_kmeans}\n fnames_orig: {fnames_orig}\n ')
    for i in range(len(fnames)):
        for file in fnames[i]:
            img_circle = cv2.imread(f'{MYPATH}/img_{folders[i]}/{file}', 0)
            print(file)

            circles = cv2.HoughCircles(img_circle, cv2.HOUGH_GRADIENT, 0.9, 25, param1=23, param2=25, maxRadius=25)
            circles_rounded = np.uint16(np.around(circles))  # radius: 185
            for j in circles_rounded[0, :]:
                #cv2.circle(img_kmeans, (i[0], i[1]), i[2], (255, 255, 0), 5)
                cv2.circle(img_circle, (j[0], j[1]), 2, (255, 0, 0), 3)
            cv2.imwrite(f'{MYPATH}/img_hough/{folders[i]}/{file[:-4]}_houghErrors.jpg', img_circle)


main()