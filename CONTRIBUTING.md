# Contributing

1. Install PyTorch for your platform, then `pip install -e ".[test]"`.
2. Run `pytest` and both scripts in `examples/` before opening a pull request.
   Tests that need CUDA or optional kernels skip on machines without them.
3. Keep numerical changes separate from refactors, and say in the pull
   request whether a change can alter gradients.

Changes to `thundersync.rollout` that touch process execution or the
sandbox should describe the threat they consider; see `SECURITY.md`.
