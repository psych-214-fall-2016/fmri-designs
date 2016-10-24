""" Outlier metrics script
"""

import sys
from os.path import basename

import numpy as np

from fmri_designs import (f_for_outliers, outlier_metrics, files_for_image,
                          parse_outlier_file)


TR = 3.0


def main():
    outlier_path = sys.argv[1]
    data_path = sys.argv[2]
    outliers = parse_outlier_file(outlier_path)
    metrics = []
    for subject in range(1, 11):
        for run in (1, 2):
            img_fname, cond_fnames = files_for_image(
                data_path, '*', subject, run)
            img_base = basename(img_fname)
            if img_base not in outliers:
                continue
            print("Outliers for {}: {}".format(img_fname, outliers[img_base]))
            hrf, outs, hrf_o, mask = f_for_outliers(
                img_fname, cond_fnames, TR, outliers[img_base])
            metrics.append(outlier_metrics(hrf, outs, hrf_o, mask))
    print("HRF      Outliers HRF+drop  HRF difference")
    for hrf, outliers, hrf_o, d in metrics:
        print('{:2.5f}  {:2.5f}  {:2.5f}  {:1.6f}'.format(
            hrf, outliers, hrf_o, d))
    print('Mean                      ', np.mean(np.array(metrics)[:, -1]))


if __name__ == '__main__':
    main()
