""" Outlier metrics script
"""

import sys

from fmri_designs.metrics import print_metrics


TR = 3.0


def main():
    print_metrics(sys.argv[1], sys.argv[2], TR)


if __name__ == '__main__':
    main()
