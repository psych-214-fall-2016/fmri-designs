""" Tests for fmri_designs package
"""

import numpy as np

from fmri_designs.regressors import events2neural
from fmri_designs.tmpdirs import dtemporize

from numpy.testing import assert_array_equal

# Example onset, duration, amplitude triplets
COND_TEST = """\
10    6.0    1
20    4.0    2
24    2.0    0.1"""

# Name of file to write for testing
COND_TEST_FNAME = 'cond_test1.txt'


# Work in temporary directory
@dtemporize
def test_events2neural_simple():
    # test events2neural function
    # Write condition test file
    with open(COND_TEST_FNAME, 'wt') as fobj:
        fobj.write(COND_TEST)
    # Read it back
    times, neural = events2neural(COND_TEST_FNAME, 32, dt=2.)
    assert_array_equal(times, np.arange(0, 32, 2))
    # Expected values for tr=2, n_trs=16
    expected = np.zeros(16)
    expected[5:8] = 1
    expected[10:12] = 2
    expected[12] = 0.1
    assert_array_equal(neural, expected)
    times, neural = events2neural(COND_TEST_FNAME, 30, dt=1.)
    assert_array_equal(times, np.arange(30))
    # Expected values for tr=1, n_trs=30
    expected = np.zeros(30)
    expected[10:16] = 1
    expected[20:24] = 2
    expected[24:26] = 0.1
    assert_array_equal(neural, expected)
