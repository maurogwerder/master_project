from PIL import Image
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt


def main():
    # Define path:
    MYPATH = os.getcwd()
    print(MYPATH)

    filenames = os.listdir(MYPATH + '/img')
    for file in filenames:
        print(file)

        # read-in files
        img = cv2.imread(MYPATH + '/img/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.uint8)

        # Cut off watermark
        img = img[:2104, :, :]

        new_comp = img
        for i in range(3):
            print(i)
            new_img = cv2.GaussianBlur(img[:,:, i], (31,31), sigmaX=0)
            new_img = normalize(new_img)

            # Combine normalized channels to complete image
            new_comp[:,:, i] = new_img

            # save image of each channel
            newIm = Image.fromarray(new_img)
            new_fname = MYPATH + '/' + 'img_norm' + '/' + str(i) + '_norm_' + file
            newIm.save(new_fname)

        # Save complete image
        newIm_comp = Image.fromarray(new_comp)
        comp_fname = MYPATH + '/img_norm/comp_norm_' + file
        newIm_comp.save(comp_fname)


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    print(img_min, img_max)
    img_new = ((img-img_min) * (1/(img_max - img_min))) ** 2 * 255
    return np.asarray(img_new, dtype=np.uint8)

main()
