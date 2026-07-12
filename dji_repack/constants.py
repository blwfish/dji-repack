"""Shared constants for dji_repack -- kept in their own module so there's
exactly one source of truth for the "already processed" archive directory
name, consumed by both merge.py (writes into it) and the discovery scan
(must never walk back into it).
"""

# Leading underscore: reads as "not primary content" to a human browsing
# the staging folder.
RAW_SPLITS_DIRNAME = "_raw_splits"
