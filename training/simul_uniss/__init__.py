"""Isolated Simul-UniSS training and evaluation utilities.

The package deliberately does not modify the existing Phase1-3 sample builders,
packers, datasets, or entrypoints.  Bootstrap schedules are token based and are
tagged as pseudo alignments until audio timestamps are available.
"""

SCHEDULE_SCHEMA_VERSION = "simul_uniss_schedule_v1"
SAMPLE_SCHEMA_VERSION = "simul_uniss_sample_v1"
PACKED_SCHEMA_VERSION = "simul_uniss_packed_v1"
