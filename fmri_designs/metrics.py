""" Code for outlier metrics script
"""

from os.path import basename

import numpy as np

from fmri_designs import (f_for_outliers, outlier_metrics, get_image,
                          get_conds, parse_outlier_file)


def print_metrics(outlier_path, image_path, cond_path, tr,
                 subjects=range(1, 11), runs=(1, 2)):
    outliers = parse_outlier_file(outlier_path)
    metrics = []
    for subject in subjects:
        for run in runs:
            img_fname = get_image(image_path, '*', subject, run)
            cond_fnames = get_conds(cond_path, '*', subject, run)
            if img_fname is None or cond_fnames == []:
                raise ValueError("Cannot find image and conditions")
            img_base = basename(img_fname)
            if img_base not in outliers:
                continue
            hrf, outs, hrf_o, mask = f_for_outliers(
                img_fname, cond_fnames, tr, outliers[img_base])
            metrics.append(outlier_metrics(hrf, outs, hrf_o, mask))
    print("{:>10s}{:>10s}{:>10s} {}".format(
        'outliers', 'HRF', 'HRF+out', 'HRF+out - HRF'))
    means = np.mean(metrics, axis=0)
    for hrf, outliers, hrf_o, d in metrics:
        print('{: 10.5f}{: 10.5f}{: 10.5f}{: 10.6f}'.format(
            outliers, hrf, hrf_o, d))
    print('Means')
    print('{: 10.5f}{:20s}{: 10.6f}'.format(
        means[1], '', means[3]))
