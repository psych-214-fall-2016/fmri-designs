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
        # TR times.  The interpolator accepts a vector of x and y.
        interp = interp1d(hr_times, conv, bounds_error=False, fill_value=0)
        # Call the interpolator with the x values we want new y values for. See
        # the docstring for `scipy.interpolate.interp1d for detail.
        cols.append(interp(tr_times))
    return np.column_stack(cols)


def f_tests_3d(Y_4d, mask, X_f, X_r):
    """ F tests on 4D data, returning 3D result.

    Parameters
    ----------
    Y_4d : array shape (I, J, K, T)
        4D data, analysis (row) dimension last.
    mask : bool array shape (I, J, K)
        mask defining voxels to analyze in `Y_4d`.
    X_f : array shape (T, P_f)
        full design for F test, where P_f >= P_r
    X_r : array shape (T, P_r)
        reduced design for F tests, where P_r >= P_f

    Returns
    -------
    f_vals : array shape (I, J, K)
        volume of F test values, zero where `mask` is False.
    n_1 : int
        degrees of freedom difference between `X_f` and `X_r`.
    n_3 : int
        degrees of freedom of error for `X_f` = T - rank(X_f).
    """
    Y = Y_4d[mask].T
    f_tests_1d, nu_1, nu_2 = f_tests(Y, X_f, X_r)
    f_tests_3d = np.zeros(Y_4d.shape[:3])
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


def f_for_outliers(image_fname, cond_fnames, tr, drop_rows, drift_order=3):
    """ F tests including outlier regressors

    Parameters
    ----------
    image_fname : str
        filename for image.  Say image shape is (I, J, K, T).
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
    f_hrf : length 3 tuple
        tuple of (f_vals, n1, n2) for F test of adding HRF regressors to drift.
        where ``f_vals`` is an array shape (I, J, K) of F values, ``n1`` is
        extra degrees of freedom in design with HRF columns, ``n2`` is
        degrees of freedom due to error in design with HRF columns.
    f_outliers : length 3 tuple
        tuple of (f_vals, n1, n2) for F test of adding outlier regressors to
        HRF and drift.  where ``f_vals`` is an array shape (I, J, K) of F
        values, ``n1`` is extra degrees of freedom in design with outlier
        columns, ``n2`` is degrees of freedom due to error in design with
        outlier columns.
    f_hrf_with_outliers : length 3 tuple
        tuple of (f_vals, n1, n2) for F test of adding HRF regressors to drift
        plus outliers specified in `drop_rows`, where ``f_vals`` is an array
        shape (I, J, K) of F values, ``n1`` is extra degrees of freedom in
        design with HRF columns, ``n2`` is degrees of freedom due to error in
        design with HRF columns.
    mask : boolean array shape (I, J, K)
        True for voxels analyzed in F test, False otherwise.
    """
    data = nib.load(image_fname).get_data()
    n_trs = data.shape[-1]
    mean = data.mean(axis=-1)
    thresh = threshold_otsu(mean)
    mask = mean > thresh
    tr_times = np.arange(n_trs) * tr
    X_hrf = conds2hrf_cols(cond_fnames, tr_times)
    X_drift = poly_drift(tr_times, drift_order)
    X_outliers = deltas_at_rows(drop_rows, n_trs)
    f_hrf = f_tests_3d(data, mask, np.c_[X_hrf, X_drift], X_drift)
    X_r = np.c_[X_hrf, X_drift]
    f_outliers = f_tests_3d(data, mask, np.c_[X_outliers, X_r], X_r)
    X_r = np.c_[X_outliers, X_drift]
    f_hrf_with_outliers = f_tests_3d(data, mask, np.c_[X_hrf, X_r], X_r)
    return f_hrf, f_outliers, f_hrf_with_outliers, mask


def outlier_metrics(f_1, f_2, f_3, mask):
    """ Return mean of F inside mask for before and after F test results

    Parameters
    ----------
    f_1 : length 3 tuple
        tuple of (f_vals, n1, n2) for F test where ``f_vals`` is an array shape
        (I, J, K) of F values, ``n1`` is extra degrees of freedom in full
        design, ``n2`` is degrees of freedom due to error in full design.
    f_2 : length 3 tuple
        tuple of same format as `f_1`.
    f_2 : length 3 tuple
        tuple of same format as `f_1`.
    mask : boolean array shape (I, J, K)
        True for voxels analyzed in F test, False otherwise.
    """
    n = np.sum(mask)
    f_1_mean = f_1[0].sum() / n
    f_2_mean = f_2[0].sum() / n
    f_3_mean = f_3[0].sum() / n
    return f_1_mean, f_2_mean, f_3_mean, f_3_mean - f_1_mean
