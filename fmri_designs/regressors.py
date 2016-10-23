""" Functions to work with standard OpenFMRI stimulus files

The functions have docstrings according to the numpy docstring standard - see:

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""

import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d

from skimage.filters import threshold_otsu
import nibabel as nib

from .spm_funcs import spm_hrf


def events2neural(task_fname, duration, dt=0.1):
    """ Return predicted neural time course from event file `task_fname`

    Predicted time course is length `duration` seconds, divided into `dt`
    second intervals.

    Parameters
    ----------
    task_fname : str or file-like
        filename of event file.
    duration : float
        duration for time course in seconds.
    dt : float, optional
        distance in seconds between two neighboring time points in output time
        course.

    Returns
    -------
    times_time_course : array shape (2, n)
        array shape (2, n) where ``n = np.ceil(duration / dt)``.  First row
        gives times in seconds for each element in second row.  Second row is
        predicted neural time course, for times given by corresponding element
        in first row.
    """
    task = np.loadtxt(task_fname)
    dt = float(dt)
    # Check that the file is plausibly a task file
    if task.ndim != 2 or task.shape[1] != 3:
        raise ValueError("Is {0} really a task file?", task_fname)
    n = int(np.ceil(duration / dt))
    ttc = np.zeros((2, n))
    # Times row
    ttc[0] = np.arange(n) * dt
    # Convert onsets, durations to dt units.
    onsets, durations, amplitudes = task.T
    onsets = np.round(onsets / dt).astype(int)
    durations = np.round(durations / dt).astype(int)
    for onset, duration, amplitude in zip(onsets, durations, amplitudes):
        ttc[1, onset:onset + duration] = amplitude
    return ttc


def poly_drift(times, order=3):
    """ Return design columns modeling polynomial drift over time

    Parameters
    ----------
    times : array length N
        times at which scans have been taken.
    order : int, optional
        order of polynomial drift

    Returns
    -------
    drift_design : array shape (N, order + 1)
        design matrix modeling polynomial drift.  Columns ordered from higher
        to lower order terms, with column of 1 at the right, for order 0.
        Except for 0-order column, columns are vector length 1.
    """
    times = np.array(times).astype(float)
    N = len(times)
    linear = times - times.mean()
    design = np.ones((N, order + 1))
    for order in range(1, order + 1):
        col = linear ** order
        col -= col.mean()
        design[:, order] = col / np.sqrt(np.sum(col ** 2))
    return np.fliplr(design)


def deltas_at_rows(rows, N):
    """ Design with columns containing single 1 at given row positions.

    Parameters
    ----------
    rows : sequence length R
        sequence of row indices.
    N : int
        number of rows in returned design.

    Returns
    -------
    delta_design : array shape (N, R)
        design matrix for modeling the effects of observations at rows given in
        `rows`.   For each ``row`` in `rows`, `delta_design` has a column of
        all zeros except for a single 1 at row index ``row``.
    """
    R = len(rows)
    delta_design = np.zeros((N, R))
    delta_design[rows, list(range(R))] = 1
    return delta_design


def spm_hrf_dt(duration=30, dt=0.1):
    """ Return spm_hrf sampled at times implied by `duration` and `dt`

    Parameters
    ----------
    duration : float, optional
        duration in seconds over which to sample HRF.
    dt : float, optional
        distance between samples in time.

    Returns
    -------
    hrf_sampled : array
        HRF sampled at ``np.arange(0, duration, dt)``.
    """
    return spm_hrf(np.arange(0, duration, dt))


def conds2hrf_cols(cond_fnames, tr_times):
    """ Return design columns for HRF-convolved neural time courses

    Parameters
    ----------
    cond_fnames : sequence
        Length C sequence of condition filenames.
    tr_times: sequence
        Length T sequence of times in seconds at which each row has been
        sampled.

    Returns
    -------
    hrf_cols : array shape (T, C)
        Columns for HRF regressors corresponding to onsets, durations,
        amplitudes in `cond_fnames`.
    """
    # Estimate to a little past the last tr_time
    duration = np.max(tr_times) + np.diff(tr_times)[-1]
    hrf = spm_hrf_dt()
    cols = []
    for cond_fname in cond_fnames:
        # Get the neural time course, sampled at default dt
        hr_times, neural = events2neural(cond_fname, duration)
        # Convolve with HRF, and drop extra HRF tail
        conv = np.convolve(neural, hrf)[:len(neural)]
        # The dt sample times may not exactly correspond to the the tr_times -
        # make a linear interpolator to sample the high-res dt samples at the
        # TR times.
        interp = interp1d(hr_times, conv, bounds_error=False, fill_value=0)
        cols.append(interp(tr_times))
    return np.column_stack(cols)


def compile_design(cond_fnames, tr, n_trs, drift_order=3):
    """ Compile design with condition `cond_fnames` and polynomial drift

    Parameters
    ----------
    cond_fnames : sequence
        Length C sequence of condition filenames.
    tr : float
        TR in seconds.
    n_trs : int
        Number of TRs.
    drift_order : int, optional
        Order of polynomial drift.

    Returns
    -------
    design : array shape ('n_trs`, C + `drift_order` + 1)
        design matrix with HRF columns first, drift columns following.  Vector
        of ones is last.
    """
    tr_times = np.arange(n_trs) * tr
    hrf_cols = conds2hrf_cols(cond_fnames, tr_times)
    drift = poly_drift(tr_times, drift_order)
    return np.c_[hrf_cols, drift]


def f_for_outliers(image_fname, cond_fnames, tr, drop_rows, drift_order=3):
    """ F tests including outlier regressors

    Parameters
    ----------
    image_fname : str
        filename for image.
    cond_fnames : sequence of str
        list of filenames for condition files.
    tr : float
        TR
    drop_rows : sequence
        rows to drop.
    drift_order, int, optional
        order for polynomial drift time.

    Returns
    -------
    f_test_3d : array
        3D array of F test values.
    nu_1 : float
        degrees of freedom difference between full and reduced design.
    nu_2 : float
        degrees of freedume due to error for full design (including outlier
        regressors).
    """
    data = nib.load(image_fname).get_data()
    n_trs = data.shape[-1]
    mean = data.mean(axis=-1)
    thresh = threshold_otsu(mean)
    mask = mean > thresh
    X_r = compile_design(cond_fnames, tr, n_trs, drift_order)
    X_extra = deltas_at_rows(drop_rows)
    X_f = np.c_[X_extra, X_r]
    in_data_2d = data[mask].T
    f_tests_1d, nu_1, nu_2 = f_tests(in_data_2d, X_f, X_r)
    f_tests_3d = np.zeros(mean.shape)
    f_tests_3d[mask] = f_tests_1d
    return f_tests_3d, nu_1, nu_2


def f_tests(Y, X_f, X_r):
    """ F tests on 2D data `Y`, with full design `X_f`, reduced `X_r`

    Parameters
    ----------
    Y : array shape (N, P)
        data, with F test value returned for each row.
    X_f : array shape (N, M1)
        full design.
    X_r : array shape (N, M2)
        reduced design.

    Returns
    -------
    f_values : array shape (P,)
        F test values for each column of `Y`.
    nu_1 : float
        extra degrees of freedom for `X_f` compared to `X_r`.
    nu_2 : float
        degrees of freedom due to error for `X_f`.
    """
    B_r = npl.pinv(X_r).dot(Y)
    rank_r = npl.matrix_rank(X_r)
    E_r = Y - X_r.dot(B_r)
    B_f = npl.pinv(X_f).dot(Y)
    rank_f = npl.matrix_rank(X_f)
    E_f = Y - X_f.dot(B_f)
    SSR_r = np.sum(E_r ** 2, axis=0)
    SSR_f = np.sum(E_f ** 2, axis=0)
    nu_1 = rank_f - rank_r
    nu_2 = X_f.shape[0] - rank_f
    return (SSR_r - SSR_f) / nu_1 / (SSR_f / nu_2), nu_1, nu_2
