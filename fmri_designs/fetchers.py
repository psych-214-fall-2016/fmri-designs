""" Fetch files for images
"""

from os.path import join as pjoin
from glob import glob
import re


def files_for_image(path, group, subject, run):
    group = group if isinstance(group, str) else '{:02d}'.format(group)
    glob_root = pjoin(path,
                      'group{}_sub{:02d}_run{}'.format(
                          group, subject, run))
    image = glob(glob_root + '.nii')
    assert len(image) <= 1
    conds = sorted(glob(glob_root + '_cond*.txt'))
    return image[0] if len(image) == 1 else None, conds


std_match = re.compile(r'group\d\d_sub\d\d_run\d\.nii\s+(\d+,\s+)*')


def parse_outlier_file(outlier_fname):
    """ Parse outlier filename `outlier_fname` for outlier indices.
    """
    outliers = {}
    with open(outlier_fname, 'rt') as fobj:
        for line in fobj:
            line = line.strip()
            if not line.startswith('group'):
                continue
            # Split on whitespace
            parts = line.split()
            if len(parts) < 2:
                continue
            rows = [int(n) for n in re.findall('[\[ ](\d+)', line)]
            outliers[parts[0]] = rows
    return outliers
