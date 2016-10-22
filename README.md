# Designs code

Code for GLM analysis of diagnostics (and other) data.

This file is in [Markdown
format](http://daringfireball.net/projects/markdown), and should render nicely
on the Github front page for this repository.

## Install

To install the necessary code:

    # Install required packages
    pip3 install --user -r requirements.txt
    # Put fmri_designs onto Python path using setup.py
    pip3 install --user --editable .

## Test

To run tests:

* install `pytest` with ``pip3 install --user pytest``;
* run tests with:

    py.test fmri_designs
