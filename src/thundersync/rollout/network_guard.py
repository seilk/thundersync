"""Install a no-network seccomp filter, then execute a command.

This helper is a separate process rather than a ``preexec_fn`` because rollout
commands are launched from worker threads.  The filter is inherited across
``execve`` and by every descendant, so udocker/PRoot cannot restore IPv4,
IPv6, packet, or netlink sockets for a policy command.  Local Unix sockets stay
available for test runners.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import socket
import sys


SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000
SCMP_CMP_NE = 1
PR_SET_NO_NEW_PRIVS = 38


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def install_network_none_filter() -> None:
    """Deny every socket family except AF_UNIX for this process tree."""
    library = ctypes.util.find_library("seccomp")
    if library is None:
        raise RuntimeError("libseccomp is required for udocker network isolation")
    seccomp = ctypes.CDLL(library, use_errno=True)
    libc = ctypes.CDLL(None, use_errno=True)

    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        value = ctypes.get_errno()
        raise OSError(value, os.strerror(value))

    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]

    context = seccomp.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        comparison = _ScmpArgCmp(0, SCMP_CMP_NE, socket.AF_UNIX, 0)
        deny = SCMP_ACT_ERRNO | errno.EPERM
        for name in (b"socket", b"socketpair"):
            syscall = seccomp.seccomp_syscall_resolve_name(name)
            if syscall < 0:
                raise RuntimeError(f"could not resolve seccomp syscall {name!r}")
            result = seccomp.seccomp_rule_add_array(
                context, deny, syscall, 1, ctypes.byref(comparison)
            )
            if result != 0:
                raise RuntimeError(
                    f"could not add seccomp rule for {name.decode()}: {result}"
                )
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"seccomp_load failed: {result}")
    finally:
        seccomp.seccomp_release(context)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m thundersync.rollout.network_guard COMMAND ...")
    install_network_none_filter()
    os.execvp(sys.argv[1], sys.argv[1:])
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
