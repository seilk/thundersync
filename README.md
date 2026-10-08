# ThunderSync

[Paper](https://arxiv.org/abs/2610.05935) · [Blog](https://seilk.github.io/thundersync)

ThunderSync is a Python library for training language-model agents while their
rollouts are still in progress. It provides streaming backward execution for
group-relative policy optimization (GRPO) and on-policy distillation (OPD).

In GRPO, each completed trajectory can contribute a gradient before the rest
of its group finishes. In OPD, each generated action can contribute as soon as
its teacher scores arrive. Parameters stay fixed during a logical batch; the
optimizer runs after all required work has completed.

## Install

ThunderSync runs on Linux with Python 3.12 or newer. The rollout sandboxes
need Docker or uDocker; uDocker's network guard needs libseccomp. Install
PyTorch for your platform, then:

```bash
pip install .
```

Optional extras: `train` (LoRA and checkpoint loading through `peft` and
`safetensors`), `kernels` (Triton, flash-linear-attention and the cuDNN
frontend), `rollout` (vLLM) and `test`. The FA4 attention path additionally
needs a FlashAttention build that provides `flash_attn.cute`.

The examples use small, randomly initialized Qwen3 models and require no model
download or dataset. They run on CPU or an available CUDA device:

```bash
python examples/streaming_grpo.py
python examples/streaming_opd.py
```

The examples live in the source repository, not in the installed package.
Optional CUDA kernels are probed on the executing device; availability is determined
by an actual forward/backward trial.

## Integrating a training loop

The examples show one complete optimizer update. In an application, replace
their token sequences with your rollout events and preserve these boundaries:

1. Open the prompt group and append generated tokens with their scored masks.
2. For GRPO, close a trajectory when its reward becomes available. For OPD,
   submit each action with teacher log probabilities for its scored tokens.
3. Close trajectories and groups after their dependencies finish. OPD may
   replay deferred state adjoints at trajectory closure.
4. Call `assert_safe_to_step()` before the optimizer update. Publish the
   resulting policy only after that update completes.

The GRPO objective is maximized (`maximize=True` in the optimizer); the OPD
objective is minimized.

GRPO's reward-linear objective implements one fixed-snapshot semigradient
epoch with group population-standard-deviation normalization. It is not a
multi-epoch PPO trainer. OPD uses a detached teacher/student log-probability
advantage and normalizes by the scored-token count of the batch it accumulates.

Tensor dtype, accumulation order, batch partitioning and kernel choice can
affect finite-precision results.

## Modules

| Module | Purpose |
| --- | --- |
| `thundersync.engine` | Streaming execution, causal segment forests, boundary adjoints and optimizer sharding. |
| `thundersync.grpo` | GRPO configuration, the reward-linear streaming objective and its data-parallel trainer. |
| `thundersync.opd` | Action-ready on-policy distillation objectives. |
| `thundersync.scheduling` | Source admission and cross-rank work claims. |
| `thundersync.rollout` | Agent rollout, bounded tool output and Docker/uDocker sandboxes. |
| `thundersync.accel` | Attention, log-probability, normalization and LoRA kernels. |

Multi-process training uses PyTorch distributed. Applications supply their
own model, data, rollout service and device allocation.

## Environment variables

| Variable | Effect |
| --- | --- |
| `THUNDERSYNC_FA4` | Streaming attention kernel: `auto` (default), `require` or `off`. |
| `THUNDERSYNC_SANDBOX_SCRATCH` | Host scratch root for sandbox working trees; must be owned by the current user. |
| `THUNDERSYNC_DOCKER_BIN`, `THUNDERSYNC_UDOCKER_BIN` | Container CLI to invoke. |
| `THUNDERSYNC_DOCKER_SEED_DIR` | Where unpacked image working trees are cached; must be owned by the current user. |
| `UDOCKER_DIR`, `PROOT_TMP_DIR` | uDocker's image store and PRoot's temporary directory. |
| `THUNDERSYNC_OPERATOR_PROFILE_RANGES` | Profiler ranges around kernels: `off` (default), `kineto`, `nvtx` or `both`. |
| `THUNDERSYNC_GRPO_CLOSE_TRACE` | Record per-pack timing of the GRPO group close. |
| `THUNDERSYNC_GRPO_PROFILE_*` | Capture a `torch.profiler` window around selected close packs. |

The profiling variables are diagnostics and change no result.

## Citation

```bibtex
@article{kang2026thundersyncrl,
  title   = {ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning},
  author  = {Kang, Seil and Kang, Hangoo and Suresh, Tarun and Kim, Youngeun and
             Pimpalgaonkar, Shreyas and Hwang, Seong Jae and Mirhoseini, Azalia},
  journal = {arXiv preprint arXiv:2610.05935},
  year    = {2026}
}
```

## License

Apache-2.0.
