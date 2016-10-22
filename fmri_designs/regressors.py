""" Functions to work with standard OpenFMRI stimulus files

The functions have docstrings according to the numpy docstring standard - see:

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""

import numpy as np

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
    ttc[0] = np.arange(n) * dt
    # Convert onsets, durations to dt units.
    onsets, durations = np.round(task[:, :2] / dt).astype(int).T
    for onset, duration, amplitude in zip(onsets, durations, task[:, 2]):
        ttc[1, onset:onset + duration] = amplitude
    return ttc
