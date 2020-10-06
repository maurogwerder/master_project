import numpy as np
import os
import cv2
import pandas as pd


def main():
    """
    Applies Hough-transform for circles to all images and saves:
     - Images highlighting hits
     - Images trying to highlight false positives
     - csv-files with coordinates of all hits
     - csv-file summarizing the total amount of hits for each file
    :return: None
    """

    # Reads-in all important file-names
    MYPATH = os.getcwd()
    fnames_kmeans = os.listdir(MYPATH + '/img_kmeans/unmasked')
    fnames_norm = os.listdir(MYPATH + '/img_norm')

    # The masks act as a ROI created by only take the clusters of kmeans-clustering with a
    # value low enough to be realistic
    fnames_masks = os.listdir(MYPATH + '/img_kmeans/masks')

    folders = ['kmeans/unmasked', 'norm', 'orig']
    fnames = [fnames_kmeans, fnames_norm]

    print(f'fnames_norm: {fnames_norm}\n fnames_kmeans: {fnames_kmeans}\n fnames_masks: {fnames_masks}\n ')
    list_of_files = None  # will hold summary of the total amount of hits for each file
    for i in range(len(fnames)):
        for file in fnames[i]:
            img_circle = cv2.imread(f'{MYPATH}/img_{folders[i]}/{file}', 0)

            # Copy file for visualization of false-positive errors
            img_error = np.copy(img_circle)
            img_mask = cv2.imread(f'{MYPATH}/img_kmeans/masks/{fnames_masks[i]}', 0)

            # Applies houghtransform to image
            circles = cv2.HoughCircles(img_circle, cv2.HOUGH_GRADIENT, 0.9, 25, param1=23, param2=25, maxRadius=25)
            circles_rounded = np.uint16(np.around(circles))

            arr_of_hits = None  # Will hold coordinates and radius of each hit per file
            for j in circles_rounded[0, :]:

                # Only include hit if the mask is not covering it
                if img_mask[j[1], j[0]] == 0:
                    cv2.circle(img_circle, (j[0], j[1]), 2, (255, 0, 0), 3)
                    cv2.circle(img_error, (j[0], j[1]), 2, (0, 0, 0), 3)

                    if arr_of_hits is None:
                        arr_of_hits = np.array((j[0], j[1], j[2])).reshape(1, 3)
                    else:
                        line = np.array((j[0], j[1], j[2])).reshape(1, 3)
                        arr_of_hits = np.concatenate((arr_of_hits, line))

            if list_of_files is None:
                list_of_files = [[file, str(len(arr_of_hits))]]
            else:
                list_of_files.append([file, str(len(arr_of_hits))])

            # Saves all files
            df_image = pd.DataFrame(arr_of_hits, columns=['x', 'y', 'radius'])
            df_image.to_csv(f'{MYPATH}/csv_files/{file[:-4]}.csv')

            df_complete = pd.DataFrame(list_of_files, columns=['file_name', 'num_of_hits'])
            df_complete.to_csv(f'{MYPATH}/csv_files/all_files.csv')
            cv2.imwrite(f'{MYPATH}/img_hough/{folders[i]}/{file[:-4]}_hough.jpg', img_circle)
            cv2.imwrite(f'{MYPATH}/img_hough/{folders[i]}/{file[:-4]}_houghErrors.jpg', img_error)



main()
