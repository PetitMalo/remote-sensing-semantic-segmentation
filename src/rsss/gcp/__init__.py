"""Support for running training on Google Cloud.

:mod:`rsss.gcp.submit` runs locally: it archives the current commit, renders the
VM's startup script and creates a spot L4 instance. :mod:`rsss.gcp.job` runs on
that instance: it stages the dataset, resumes from GCS, calls :mod:`rsss.train`
and synchronises the run directory back to the bucket while training.
"""
