"""
Remote MD Dispatch

Runs CRM1 docking MD for a single NES candidate on a remote Linux machine
over SSH, instead of running it locally in this process.

This is entirely opt-in and config-driven via environment variables - if
MD_REMOTE_HOST / MD_REMOTE_USER aren't set, REMOTE_ENABLED is False and
md_job_queue.py falls back to running everything locally, exactly as before.

No credentials are stored in source. Remote auth MUST be SSH key-based
(BatchMode=yes is used below, which fails fast instead of hanging on a
password prompt) - see "MD files/REMOTE_MD_SETUP.md" for how to set that up.

The SSH key used for this should be a RESTRICTED key (forced to run only
remote_md_gatekeeper.sh on the remote side, via a "command=" entry in that
machine's authorized_keys) - see the setup doc. That way, even if this key
is copied or stolen, it can only touch job_*.json files inside
MD_REMOTE_WORKDIR, not the rest of the remote filesystem.

MD_REMOTE_WORKDIR MUST be an absolute path (not "~/...") when using a
restricted key, since the gatekeeper script matches against the exact
literal command string it receives, and tilde expansion would not happen
consistently between this file and the remote shell.

Environment variables:
    MD_REMOTE_HOST     Remote hostname, e.g. abc123-pc12345.bioc.private.cam.ac.uk
    MD_REMOTE_USER     Remote username, e.g. ob419@cam.ac.uk
    MD_REMOTE_KEY      Path to the SSH private key to use for non-interactive auth
    MD_REMOTE_WORKDIR  Absolute working directory on the remote machine
                        (e.g. /home/ob419/md_worker)
    MD_REMOTE_PYTHON   Python interpreter on the remote machine, e.g. the conda
                        env's python (default: "python3")
"""

import os
import json
import uuid
import subprocess
import tempfile
from pathlib import Path

REMOTE_HOST = os.environ.get('MD_REMOTE_HOST')
REMOTE_USER = os.environ.get('MD_REMOTE_USER')
REMOTE_KEY = os.environ.get('MD_REMOTE_KEY')
REMOTE_PORT = os.environ.get('MD_REMOTE_PORT', '22')
REMOTE_WORKDIR = os.environ.get('MD_REMOTE_WORKDIR', '~/md_worker')
REMOTE_PYTHON = os.environ.get('MD_REMOTE_PYTHON', 'python3')

REMOTE_ENABLED = bool(REMOTE_HOST and REMOTE_USER)


def _ssh_base():
    cmd = ['ssh']
    if REMOTE_KEY:
        cmd += ['-i', REMOTE_KEY]
    # BatchMode=yes: never prompt for a password - fail immediately instead,
    # so a broken/expired key surfaces as a fast exception rather than a hang.
    cmd += ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new']
    # ssh uses lowercase -p for a non-default port (e.g. RunPod's "SSH over
    # exposed TCP" option, which uses a random high port rather than 22).
    cmd += ['-p', REMOTE_PORT]
    cmd += ['-l', REMOTE_USER, REMOTE_HOST]
    return cmd


def _scp_base():
    cmd = ['scp']
    if REMOTE_KEY:
        cmd += ['-i', REMOTE_KEY]
    cmd += ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new']
    # scp uses uppercase -P for the port - a different flag letter than ssh,
    # easy to get wrong.
    cmd += ['-P', REMOTE_PORT]
    return cmd


def check_remote_ready(timeout_sec: int = 15) -> bool:
    """
    Quick connectivity + setup check. Verifies SSH works without a password
    prompt and that remote_md_runner.py exists in the remote workdir.
    Intended to be called once at server startup, not per-job.
    """
    if not REMOTE_ENABLED:
        return False

    try:
        result = subprocess.run(
            _ssh_base() + [f'test -f {REMOTE_WORKDIR}/remote_md_runner.py && echo OK'],
            capture_output=True, text=True, timeout=timeout_sec
        )
        return result.returncode == 0 and 'OK' in result.stdout
    except Exception:
        return False


def run_remote_docking(pdb_content: str, candidate: dict, duration_ns: float,
                        timeout_sec: int = 3600) -> dict:
    """
    Run CRM1 docking MD for a single candidate on the remote machine and
    return the enhanced candidate dict (same shape NESMDRefiner would
    produce locally).

    Raises on any failure (SSH, remote script error, timeout). The caller
    (md_job_queue._run_job_thread) already wraps per-candidate processing
    in a try/except and falls back to local execution on failure, so this
    function intentionally does not swallow errors itself.
    """
    if not REMOTE_ENABLED:
        raise RuntimeError("Remote MD is not configured (MD_REMOTE_HOST/MD_REMOTE_USER unset)")

    job_tag = uuid.uuid4().hex[:8]
    remote_input = f'{REMOTE_WORKDIR}/job_{job_tag}_input.json'
    remote_output = f'{REMOTE_WORKDIR}/job_{job_tag}_output.json'

    with tempfile.TemporaryDirectory() as tmpdir:
        local_input = Path(tmpdir) / f'job_{job_tag}_input.json'
        local_output = Path(tmpdir) / f'job_{job_tag}_output.json'

        local_input.write_text(json.dumps({
            'pdb_content': pdb_content,
            'candidate': candidate,
            'duration_ns': duration_ns
        }))

        # 1. Make sure the remote workdir exists
        subprocess.run(_ssh_base() + [f'mkdir -p {REMOTE_WORKDIR}'],
                        check=True, timeout=30)

        # 2. Copy the job input up
        subprocess.run(
            _scp_base() + [str(local_input), f'{REMOTE_USER}@{REMOTE_HOST}:{remote_input}'],
            check=True, timeout=60
        )

        # 3. Run the remote docking script. "-u" (unbuffered) is essential
        # here: without it, Python fully buffers stdout whenever it isn't
        # attached to a real terminal (exactly the case over this SSH pipe,
        # which doesn't allocate a pty) - every print() in md_refinement.py
        # would sit invisible on the remote side until the buffer filled or
        # the process exited, making a perfectly healthy run look silently
        # stuck for minutes at a time.
        remote_cmd = f'{REMOTE_PYTHON} -u {REMOTE_WORKDIR}/remote_md_runner.py {remote_input} {remote_output}'
        try:
            subprocess.run(_ssh_base() + [remote_cmd], check=True, timeout=timeout_sec)
        finally:
            # 4. Copy the result back (best-effort even if the ssh call above
            #    reported a non-zero exit, in case a partial/error result was written)
            try:
                subprocess.run(
                    _scp_base() + [f'{REMOTE_USER}@{REMOTE_HOST}:{remote_output}', str(local_output)],
                    check=True, timeout=60
                )
            except Exception:
                pass

            # 5. Best-effort cleanup of remote temp files
            subprocess.run(
                _ssh_base() + [f'rm -f {remote_input} {remote_output}'],
                timeout=30
            )

        if not local_output.exists():
            raise RuntimeError("Remote MD run did not produce an output file")

        return json.loads(local_output.read_text())
