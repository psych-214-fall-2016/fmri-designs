""" Tests for regressors module.

Tests in ``py.test`` format.
"""

from os.path import dirname, join as pjoin

import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d

from skimage.filters import threshold_otsu
import nibabel as nib

from fmri_designs.regressors import (events2neural_hr, poly_drift,
                                     deltas_at_rows, spm_hrf_dt,
                                     conds2hrf_cols, f_tests, f_tests_3d,
                                     f_for_outliers, outlier_metrics)
from fmri_designs.tmpdirs import dtemporize
from fmri_designs.spm_funcs import spm_hrf

import pytest
from numpy.testing import assert_array_equal, assert_almost_equal

HERE = dirname(__file__)
CONDS = pjoin(HERE, 'conds')

# Example onset, duration, amplitude triplets
COND_TEST = """\
10    6.0    1
20    4.0    2
24    2.0    0.1"""

# Name of file to write for testing
COND_TEST_FNAME = 'cond_test1.txt'


# Work in temporary directory
@dtemporize
def test_events2neural_hr_simple():
    # test events2neural_hr function
    # Write condition test file
    with open(COND_TEST_FNAME, 'wt') as fobj:
        fobj.write(COND_TEST)
    # Read it back
    times_neural = events2neural_hr(COND_TEST_FNAME, 32, dt=2.)
    assert times_neural.shape == (2, 16)
    times, neural = times_neural
    assert_array_equal(times, np.arange(0, 32, 2))
    # Expected values for tr=2, n_trs=16
    expected = np.zeros(16)
    expected[5:8] = 1
    expected[10:12] = 2
    expected[12] = 0.1
    assert_array_equal(neural, expected)
    times, neural = events2neural_hr(COND_TEST_FNAME, 30, dt=1.)
    assert_array_equal(times, np.arange(30))
    # Expected values for tr=1, n_trs=30
    expected = np.zeros(30)
    expected[10:16] = 1
    expected[20:24] = 2
    expected[24:26] = 0.1
    assert_array_equal(neural, expected)
    # Extend duration, more zeros in neural
    times, neural = events2neural_hr(COND_TEST_FNAME, 40, dt=1.)
    assert_array_equal(times, np.arange(40))
    assert_array_equal(neural, np.concatenate((expected, np.zeros(10))))
    # Drop duration, truncates
    times, neural = events2neural_hr(COND_TEST_FNAME, 20, dt=1.)
    assert_array_equal(times, np.arange(20))
    assert_array_equal(neural, expected[:20])
    # dt of 0.1
    times, neural = events2neural_hr(COND_TEST_FNAME, 30, dt=0.1)
    assert_array_equal(times, np.arange(0, 30, 0.1))
    # Expected values for tr=0.1, n_trs=300
    expected = np.zeros(300)
    expected[100:160] = 1
    expected[200:240] = 2
    expected[240:260] = 0.1
    assert_array_equal(neural, expected)
    # 0.1 is the default
    times, neural = events2neural_hr(COND_TEST_FNAME, 30)
    assert_array_equal(times, np.arange(0, 30, 0.1))
    assert_array_equal(neural, expected)


def test_poly_drift():
    # Test polynomial drift
    times = np.arange(0, 20, 2.15)
    n = len(times)
    assert_array_equal(poly_drift(times, 0), np.ones((n, 1)))
    linear = np.linspace(-1, 1, n)
    linear /= npl.norm(linear)
    assert_almost_equal(poly_drift(times, 1), np.c_[linear, np.ones(n)])
    quadratic = (linear ** 2) - (linear ** 2).mean()
    quadratic /= npl.norm(quadratic)
    assert_almost_equal(poly_drift(times, 2),
                        np.c_[quadratic, linear, np.ones(n)])
    cubic = (linear ** 3) - (linear ** 3).mean()
    cubic /= npl.norm(cubic)
    assert_almost_equal(poly_drift(times, 3),
                        np.c_[cubic, quadratic, linear, np.ones(n)])
    d_10 = poly_drift(times, 10)
    exp_sums = np.concatenate((np.zeros(10), [n]))
    assert_almost_equal(np.sum(d_10, axis=0), exp_sums)
    exp_lengths = np.concatenate((np.ones(10), [np.sqrt(n)]))
    assert_almost_equal(np.sqrt(np.sum(d_10 ** 2, axis=0)), exp_lengths)


def test_deltas_at_rows():
    # Test design for deltas as given rows
    for rows, n in (([3, 7, 15], 20),
                    ([1, 2, 9], 12)):
        d = deltas_at_rows(rows, n)
        for col, row in enumerate(rows):
            assert d[row, col] == 1
            d[row, col] = 0
        assert np.all(d == 0)
    with pytest.raises(IndexError):
        deltas_at_rows([3, 12], 12)


def test_spm_hrf_dt():
    # Test SPM HRF at duration and dt
    assert_almost_equal(spm_hrf(np.arange(0, 30, 0.1)),
                        spm_hrf_dt())
    assert_almost_equal(spm_hrf(np.arange(0, 20, 0.1)),
                        spm_hrf_dt(20))
    assert_almost_equal(spm_hrf(np.arange(10)),
                        spm_hrf_dt(10, 1))


def test_conds2hrf_cols():
    # Test function to return design columns for condition files
    cond_fnames = [pjoin(HERE, 'ds114_sub009_t2r1_cond.txt'),
                   pjoin(HERE, 'new_cond.txt')]
    TR = 2.5
    tr_times = np.arange(0, 400, TR)
    hrf_cols = conds2hrf_cols(cond_fnames, tr_times)
    # Now go the slow way round
    hrf = spm_hrf(np.arange(0, 30, 0.1))
    hr_times, neural = events2neural_hr(cond_fnames[0], 410)
    conv = np.convolve(neural, hrf)[:len(neural)]
    interp0 = interp1d(hr_times, conv, bounds_error=False, fill_value=0)
    hr_times, neural = events2neural_hr(cond_fnames[1], 410)
    conv = np.convolve(neural, hrf)[:len(neural)]
    interp1 = interp1d(hr_times, conv, bounds_error=False, fill_value=0)
    hrf_cols_manual = np.c_[interp0(tr_times), interp1(tr_times)]
    assert_almost_equal(hrf_cols, hrf_cols_manual)


def test_f_tests():
    # Test F test routine against results from R
    # See f_tests.R
    x = np.loadtxt(pjoin(HERE, 'x.txt'))
    y1 = np.loadtxt(pjoin(HERE, 'y1.txt'))
    y2 = np.loadtxt(pjoin(HERE, 'y2.txt'))
    Y = np.c_[y1, y2]
    n = len(x)
    X_f = np.ones((n, 2))
    X_f[:, 1] = x
    F, nu_1, nu_2 = f_tests(Y, X_f, np.ones((n, 1)))
    # Test F test results come from the output of R
    exp_f = [0.01197, 0.5955]
    assert_almost_equal(F, exp_f, 4)
    assert (nu_1, nu_2) == (1, 98)
    # Test 3D version
    vol_shape = (2, 3, 4)
    Y_4d = np.zeros(vol_shape + (n,))
    mask = np.zeros(vol_shape, dtype=bool)
    Y_4d[1, 1, 1] = y1
    Y_4d[1, 1, 2] = y2
    mask[1, 1, 1] = True
    mask[1, 1, 2] = True
    exp_f_3d = np.zeros(vol_shape, dtype=float)
    exp_f_3d[mask] = exp_f
    F, nu_1, nu_2 = f_tests_3d(Y_4d, mask, X_f, np.ones((n, 1)))
    assert_almost_equal(F, exp_f_3d, 4)
    assert (nu_1, nu_2) == (1, 98)


def test_f_for_outliers():
    # More or less smoke test for big-picture routine
    img_fname = pjoin(HERE, 'group00_sub04_run1.nii')
    cond_fnames = [pjoin(CONDS, 'group00_sub04_run1_cond1.txt'),
                   pjoin(CONDS, 'group00_sub04_run1_cond2.txt'),
                   pjoin(CONDS, 'group00_sub04_run1_cond3.txt'),
                   pjoin(CONDS, 'group00_sub04_run1_cond4.txt')]
    tr = 2.0
    f_1, f_2, f_3, msk = f_for_outliers(img_fname, cond_fnames, tr, [2, 3])
    # Long manual process
    data = nib.load(img_fname).get_data()
    n_trs = data.shape[-1]
    tr_times = np.arange(n_trs) * tr
    hrf_cols = conds2hrf_cols(cond_fnames, tr_times)
    drift = poly_drift(tr_times, 3)
    outliers = deltas_at_rows([2, 3], n_trs)
    mean = data.mean(axis=-1)
    mask = mean > threshold_otsu(mean)
    assert_array_equal(mask, msk)
    f_hrf, h_n1, h_n2 = f_tests_3d(data, mask, np.c_[hrf_cols, drift], drift)
    assert_array_equal(f_1[0], f_hrf)
    reduced = np.c_[hrf_cols, drift]
    f_outliers, o_n1, o_n2 = f_tests_3d(data, mask,
                                        np.c_[outliers, reduced], reduced)
    reduced = np.c_[outliers, drift]
    f_both, b_n1, b_n2 = f_tests_3d(data, mask,
                                    np.c_[hrf_cols, reduced], reduced)
    assert_array_equal(f_2[0], f_outliers)
    assert_array_equal(f_3[0], f_both)
    # HRF cols 2 and 3 all zero because of truncated data (first 10 vols)
    assert f_1[1:] == (2, n_trs - 6)
    assert f_2[1:] == (2, n_trs - 8)
    assert f_3[1:] == (2, n_trs - 8)


def test_outlier_metrics():
    a = np.random.normal(10, 2, size=(10, 11, 12))
    b = np.random.normal(10, 2, size=(10, 11, 12))
    c = np.random.normal(10, 2, size=(10, 11, 12))
    msk = np.random.normal(10, 2, size=(10, 11, 12)) > 0.2
    n = np.sum(msk)
    first, second, third, d = outlier_metrics((a, 1, 1), (b, 1, 1), 
                                              (c, 1, 1), msk)
    assert first == a[msk].sum() / n
    assert second == b[msk].sum() / n
    assert third == c[msk].sum() / n
    assert d == third - first
