"""Run model Python in a private Linux filesystem, PID and network namespace.

Only the interpreter, shared libraries, read-only EHR database and the task's
private workspace are mounted. Failure to create isolation never falls back
to executing model code on the host.
"""
import ctypes
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time


RUNNER = r'''
import json, os, runpy, sys
with open('/policy.json') as f:
    policy = json.load(f)
message = 'ILLEGAL_OPERATION: Access to this path is prohibited. Use the provided EHR database to answer the question.'
def allowed(path):
    if isinstance(path, int) or path is None:
        return True
    path = os.path.realpath(os.fsdecode(path))
    return path in policy['files'] or any(path == base or path.startswith(base + '/') for base in policy['directories'])
def audit(event, args):
    if event in ('open', 'os.listdir', 'os.scandir', 'os.chdir') and args and not allowed(args[0]):
        os.write(2, (message + '\n').encode())
        raise PermissionError(message)
sys.addaudithook(audit)
if not os.path.lexists('/work/data'):
    os.symlink('/data', '/work/data')
import duckdb
original_connect = duckdb.connect
def connect(database=':memory:', read_only=False, config=None):
    options = dict(config or {})
    options.pop('allowed_directories', None)
    options.update(enable_external_access=True, lock_configuration=False, threads=32)
    connection = original_connect(database, read_only=read_only, config=options)
    # DuckDB requires the database to be started before setting this allowlist.
    if not connection.execute("SELECT current_setting('lock_configuration')").fetchone()[0]:
        connection.execute("SET allowed_directories = ['/tmp', '/work']")
        connection.execute('SET enable_external_access = false')
        connection.execute('SET lock_configuration = true')
    return connection
duckdb.connect = connect
duckdb.execute("SET allowed_directories = ['/tmp', '/work']")
duckdb.execute('SET enable_external_access = false')
duckdb.execute('SET threads = 32')
duckdb.execute('SET lock_configuration = true')
try:
    runpy.run_path('/action.py', run_name='__main__')
except duckdb.PermissionException as exc:
    raise PermissionError(message) from exc
'''


def _clean_environment():
    env = {
        'PATH': str(Path(sys.executable).parent) + ':/usr/bin:/bin',
        'HOME': '/work', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    }
    return env


def run_isolated(code_file, timeout=60, workspace=None):
    start = time.monotonic()
    project = Path(__file__).resolve().parents[3]
    database = Path(os.environ.get('EHR_COMPLEX_DB', str(project / 'data/mimic_iv_full.duckdb'))).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='ehrcomplex-isolated-') as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        (root / 'work').mkdir()
        (root / 'tmp').mkdir(mode=0o1777)
        (root / 'action.py').write_bytes(Path(code_file).read_bytes())
        (root / 'runner.py').write_text(RUNNER)
        command = [
            'unshare', '--user', '--map-root-user', '--mount', '--net', '--ipc',
            '--pid', '--fork',
            sys.executable, '-I', '-B', str(Path(__file__).resolve()),
            '--child', str(root), str(database), str(project / 'data/mimic_iv_full.duckdb'),
            str(workspace or ''),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=_clean_environment(), close_fds=True,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return -1, stdout, stderr + f'\nProcess timed out after {timeout}s; isolated processes terminated.', time.monotonic() - start
        return process.returncode, stdout, stderr, time.monotonic() - start


def _child(root, database, database_alias, workspace=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: parent-death signal')
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    libc.mount.restype = ctypes.c_int
    def mount(source, destination, flags):
        source = os.fsencode(source) if source else None
        if libc.mount(source, os.fsencode(destination), None, flags, None):
            raise OSError(ctypes.get_errno(), f'ISOLATION_SETUP_FAILED: mount refused ({source!r}, {destination}, {flags})')
    # Do not propagate mounts into the host or other evaluations.
    mount(None, '/', (1 << 18) | (1 << 14))  # MS_PRIVATE | MS_REC
    mount(str(root), str(root), 4096)  # Make the new root a mount point.
    directories, files = [], []
    def bind(source, destination=None, device=False):
        source = Path(source)
        if not source.exists():
            return
        destination = str(destination or source)
        target = root / destination.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            target.mkdir(exist_ok=True)
            directories.append(destination)
        else:
            target.touch()
            files.append(destination)
        mount(str(source), str(target), 4096 | 16384)  # MS_BIND | MS_REC (locked container submounts).
        # Remount every nested mount read-only, preserving existing locked flags.
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            fields = line.split()
            point, options = fields[4], set(fields[5].split(','))
            if point == str(target) or point.startswith(str(target) + '/'):
                flags = 4096 | 32 | 1 | 2 | (0 if device else 4)
                for name, flag in [('noexec', 8), ('noatime', 1024), ('nodiratime', 2048), ('relatime', 1 << 21)]:
                    if name in options:
                        flags |= flag
                mount(None, point, flags)

    for source in [sys.prefix, '/usr/lib', '/usr/lib64', '/lib', '/lib64', '/usr/share/zoneinfo']:
        bind(source)
    for source in ['/etc/ld.so.cache', '/etc/localtime', '/bin/sh', '/bin/bash', '/usr/bin/env']:
        bind(source)
    for source in ['/dev/null', '/dev/urandom', '/dev/random']:
        bind(source, device=True)
    for destination in set([str(database), database_alias, '/data/mimic_iv_full.duckdb', '/work/data/mimic_iv_full.duckdb']):
        bind(database, destination)
    if workspace:
        for name in ('work', 'tmp'):
            source = Path(workspace) / name
            source.mkdir(exist_ok=True)
            mount(str(source), str(root / name), 4096 | 16384)
    (root / 'policy.json').write_text(json.dumps({'directories': directories + ['/work', '/tmp'], 'files': files + ['/runner.py', '/policy.json', '/action.py']}))
    # Remove the host root mount entirely, including its parent-directory path.
    (root / '.oldroot').mkdir()
    os.chdir(root)
    pivot_root = {'x86_64': 155, 'aarch64': 41}[platform.machine()]
    if libc.syscall(pivot_root, ctypes.c_char_p(b'.'), ctypes.c_char_p(b'.oldroot')):
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: pivot_root')
    os.chdir('/')
    if libc.umount2(b'/.oldroot', 2):  # MNT_DETACH
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: detach host root')
    os.rmdir('/.oldroot')
    os.chdir('/work')
    # We are root only in the new user namespace. Drop every capability before
    # exec, lock out UID-0 capability restoration, and prevent privileged exec.
    if libc.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: no_new_privs')
    if libc.prctl(28, 3, 0, 0, 0):  # SECBIT_NOROOT | SECBIT_NOROOT_LOCKED
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: securebits')
    for capability in range(64):
        if libc.prctl(24, capability, 0, 0, 0) and ctypes.get_errno() != 22:
            raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: capability bounding set')
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Caps(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
    header, caps = Header(0x20080522, 0), (Caps * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(caps)):
        raise OSError(ctypes.get_errno(), 'ISOLATION_SETUP_FAILED: capability drop')
    os.execve(sys.executable, [sys.executable, '-I', '-B', '/runner.py'], _clean_environment())


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--child':
        raise SystemExit('This is the internal isolated-code launcher.')
    _child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5] or None)
