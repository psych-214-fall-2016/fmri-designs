""" Outlier metrics script
"""

import sys

from fmri_designs.metrics import print_metrics


def main():
    outlier_path, image_path, cond_path, tr, group = sys.argv[1:6]
    print_metrics(outlier_path, image_path, cond_path, float(tr), int(group))


if __name__ == '__main__':
    main()
