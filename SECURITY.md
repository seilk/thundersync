# Security

Please report vulnerabilities privately through GitHub's
[security advisory form](../../security/advisories/new) rather than a public
issue.

## Sandboxes are not a security boundary

`thundersync.rollout` runs policy-generated commands inside Docker or uDocker
containers. These sandboxes isolate rollouts from each other for
reproducibility; they are not hardened against a hostile policy. Run rollouts
for untrusted models on a dedicated host or VM.

Docker sandboxes run as the image's default user (root unless the image sets
`USER`) with:

- no network (`--network none`; loopback only),
- `--security-opt no-new-privileges`,
- every capability dropped except `CHOWN`, `DAC_OVERRIDE`, `FOWNER`,
  `SETUID`, `SETGID` and `KILL`,
- 4 GB of memory, 2 CPUs and 512 processes by default.

uDocker sandboxes run under PRoot, which is a convenience layer, not an
isolation mechanism, and cannot enforce memory, CPU or process limits. Their
optional seccomp network guard blocks socket creation for the native
architecture only and leaves Unix-domain sockets open. A task with
`requires_network=True` runs with the host's network and without the guard.

Host-side handling:

- Scratch and seed directories must be owned by the current user and are
  created mode 0700.
- Host Git reads only a trusted snapshot taken before the policy runs, with
  system and global configuration, fsmonitor and hooks disabled.
- A pooled sandbox is reused only when its tracked files match the snapshot.
  Generated caches (`__pycache__`, `.pyc`, `.egg-info`, `.pytest_cache`) and
  other state outside the repository are not reset between uses.
