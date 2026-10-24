"""What the generation side and the runtime side share: chart identity and play style (charts.py) and the files of
the distributed DB, contract 2 (DB build to recommendation core) (dbformat.py). Both sides write a full DB with the same function, so "a full" and
"the previous full + a diff" have the same content.

This package imports nothing else from octantvector, reads no settings and uses no generation inputs or user data.
"""
