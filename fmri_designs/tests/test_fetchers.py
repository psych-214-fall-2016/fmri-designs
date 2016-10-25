""" Tests for fetchers module
"""

from os.path import dirname, join as pjoin

from fmri_designs.fetchers import fetch_image, fetch_conds, parse_outlier_file

from fmri_designs.tmpdirs import dtemporize

HERE = dirname(__file__)
THERE = pjoin(HERE, 'conds')


def test_files_for_image():
    for group in (0, '*'):
        image = fetch_image(HERE, group, 4, 1)
        assert image == pjoin(HERE, 'group00_sub04_run1.nii')
        conds = fetch_image(THERE, group, 4, 1)
        assert conds == [pjoin(HERE, 'group00_sub04_run1_cond1.txt'),
                        pjoin(HERE, 'group00_sub04_run1_cond2.txt'),
                        pjoin(HERE, 'group00_sub04_run1_cond3.txt'),
                        pjoin(HERE, 'group00_sub04_run1_cond4.txt')]
        assert fetch_image(HERE, group, 3, 1) is None
        assert fetch_conds(THERE, group, 3, 1) == []


@dtemporize
def test_parse_outlier_file():
    # Test parsing of outlier output file
    with open('outliers.txt', 'wt') as fobj:
        fobj.write("""\
group00_sub08_run2.nii 105, 106, 108, 156, 157, 158
group00_sub09_run1.nii
""")
    assert parse_outlier_file('outliers.txt') == {
        'group00_sub08_run2.nii': [105, 106, 108, 156, 157, 158]}
    with open('outliers2.txt', 'wt') as fobj:
        fobj.write("""\
group02_sub08_run2.nii outliers:[64, 105]
group02_sub09_run1.nii outliers:[0, 31, 32, 56, 88, 115, 116]
""")
    assert parse_outlier_file('outliers2.txt') == {
        'group02_sub08_run2.nii': [64, 105],
        'group02_sub09_run1.nii': [0, 31, 32, 56, 88, 115, 116]}
