""" Code for outlier metrics script
"""

from os.path import basename

import numpy as np

from fmri_designs import (f_for_outliers, outlier_metrics, files_for_image,
                          parse_outlier_file)


def print_metrics(outlier_path, data_path, tr):
    outliers = parse_outlier_file(outlier_path)
    metrics = []
    for subject in range(1, 11):
        for run in (1, 2):
            img_fname, cond_fnames = files_for_image(
                data_path, '*', subject, run)
            img_base = basename(img_fname)
            if img_base not in outliers:
                continue
            hrf, outs, hrf_o, mask = f_for_outliers(
                img_fname, cond_fnames, tr, outliers[img_base])
            metrics.append(outlier_metrics(hrf, outs, hrf_o, mask))
    print("{:>10s}{:>10s}{:>10s} {}".format(
        'outliers', 'HRF', 'HRF+out', 'HRF+out difference'))
    means = np.mean(metrics, axis=0)
    for hrf, outliers, hrf_o, d in metrics:
        print('{: 10.5f}{: 10.5f}{: 10.5f}{: 10.6f}'.format(
            outliers, hrf, hrf_o, d))
    print('Means')
    print('{: 10.5f}{:20s}{: 10.6f}'.format(
        means[1], '', means[3]))
