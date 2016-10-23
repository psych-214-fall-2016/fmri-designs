""" Fetch files for images
"""

from os.path import join as pjoin
from glob import glob


def files_for_image(path, group, subject, run):
    glob_root = pjoin(path,
                      'group{:02d}_sub{:02d}_run{}'.format(
                          group, subject, run))
    image = glob(glob_root + '.nii')
    assert len(image) <= 1
    conds = sorted(glob(glob_root + '_cond*.txt'))
    return image[0] if len(image) == 1 else None, conds
