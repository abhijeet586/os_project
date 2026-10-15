"""Event-sourced measurement.

The collector is append-only and derived statistics are computed *after* a run,
never during it. That keeps the simulation hot path free of measurement bias
and lets new questions be answered from an old event log without re-simulating.
"""
