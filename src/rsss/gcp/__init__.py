"""Support for running training on Google Cloud.

:mod:`rsss.gcp.job` is the on-instance entry point: it stages the dataset from
GCS (or from IGN on the first run), resumes from the last checkpoint, calls
:mod:`rsss.train` and synchronises the run directory back to the bucket while
training.
"""
