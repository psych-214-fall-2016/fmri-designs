""" Fetch files for images
"""

from os.path import join as pjoin
from glob import glob
import re


def fetch_image(path, group, subject, run):
    """ Get image filename from `path` for given `group`, `subject` and `run`
    """
    group = group if isinstance(group, str) else '{:02d}'.format(group)
    glob_root = pjoin(path,
                      'group{}_sub{:02d}_run{}'.format(
                          group, subject, run))
    image = glob(glob_root + '.nii')
    return image[0] if len(image) == 1 else None


def fetch_conds(path, group, subject, run):
    """ Condition filenames from `path` for given `group`, `subject` and `run`
    """
    group = group if isinstance(group, str) else '{:02d}'.format(group)
    glob_root = pjoin(path,
                      'group{}_sub{:02d}_run{}'.format(
                          group, subject, run))
    return sorted(glob(glob_root + '_cond*.txt'))


std_match = re.compile(r'group\d\d_sub\d\d_run\d\.nii\s+(\d+,\s+)*')


def parse_outlier_file(outlier_fname):
    """ Parse outlier filename `outlier_fname` for outlier indices.

    Tries to parse everyone's output format (I'm looking at you group02!).
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
