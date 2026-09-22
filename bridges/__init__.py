"""Implementations of the `wmbench` protocols for this repository.

The benchmark package ships no Ctrl-World code and no ABC code, so the two halves have to
meet somewhere: this is that place, and the only code in the repository that imports both
`models/` and `wmbench`. Keeping it in one directory keeps the dependency one-way -
`wmbench` never learns what a Ctrl-World checkpoint is, and `models/` never learns what a
benchmark run is.

Nothing here is imported by the training or rollout entry points, so a checkout without
`wmbench` installed still trains.
"""
