# init for fmri_designs package
___version__ = '0.1'

from .regressors import (events2neural, poly_drift, deltas_at_rows, spm_hrf_dt,
                         conds2hrf_cols, compile_design, f_tests,
                         f_for_outliers)

from .fetchers import files_for_image
