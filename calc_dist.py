import pandas as pd
import os
import numpy as np
from scipy.spatial import distance


def main():
    MYPATH = os.getcwd()
    fnames_csv = os.listdir(f'{MYPATH}/csv_files')

    str_dist = ""
    for file in fnames_csv:
        print(f'Currently processed: {file}')
        arr = np.asarray(pd.read_csv(f'{MYPATH}/csv_files/{file}'))
        dists_mean = None
        for i in range(len(arr)):
            dists = None
            for j in range(len(arr)):
                res = distance.euclidean(arr[i, 1:3], arr[j, 1:3])  # Calc dist of all cells to one specific cell
                if res != 0:
                    if dists is None:
                        dists = [res]
                    else:
                        dists.append(res)
            dists.sort()
            if dists_mean is None:
                dists_mean = [np.mean(dists[:6])]  # mean of distances to the 6 closest neighbouring cells
            else:
                dists_mean.append(np.mean(dists[:6]))
        str_dist += f'{file}: {np.mean(dists_mean)}\n'
    with open("distances.txt", "w") as f:
        f.write(str_dist)


main()
