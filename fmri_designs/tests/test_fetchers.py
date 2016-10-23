""" Tests for fetchers module
"""

from os.path import dirname, join as pjoin

from fmri_designs.fetchers import files_for_image

HERE = dirname(__file__)


def test_files_for_image():
    image, conds = files_for_image(HERE, 0, 4, 1)
    assert image == pjoin(HERE, 'group00_sub04_run1.nii')
    assert conds == [pjoin(HERE, 'group00_sub04_run1_cond1.txt'),
                     pjoin(HERE, 'group00_sub04_run1_cond2.txt'),
                     pjoin(HERE, 'group00_sub04_run1_cond3.txt'),
                     pjoin(HERE, 'group00_sub04_run1_cond4.txt')]
    image, conds = files_for_image(HERE, 0, 3, 1)
    assert image is None
    assert conds == []
