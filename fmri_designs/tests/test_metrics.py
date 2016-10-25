""" Test metrics module
"""

from os.path import dirname, join as pjoin

from fmri_designs.fetchers import parse_outlier_file
from fmri_designs.metrics import print_metrics

HERE = dirname(__file__)


def test_regression(capsys):
    # Test whether the output has changed from a previous run
    outlier_fname = pjoin(HERE, 'outliers.txt')
    assert parse_outlier_file(outlier_fname) == {'group00_sub04_run1.nii':
                                                 [2, 3]}
    print_metrics(outlier_fname, HERE, pjoin(HERE, 'conds'), 3.0,
                  subjects=(4,), runs=(1,))
    out, err = capsys.readouterr()
    assert out == """\
  outliers       HRF   HRF+out HRF+out - HRF
  17.71026   1.90616  15.55293 13.646775
Means
  17.71026                     13.646775
"""
