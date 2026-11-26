"""Where the code runs (Colab or an own computer with a GPU), the dataset download, and
saving the results.

Every training run writes everything to outputs/<model>/ (weights, figures, results); the LIME
notebook writes to outputs/lime/<model>/. Below, <name> is that folder relative to outputs/
("resnet50", "lime/convnext_tiny", ...). `save_results` then keeps them permanently:
  * GitHub: figures and result files (no weights) -> results/<name>/ in this repository,
    committed and pushed. Works from Colab (with a GITHUB_TOKEN secret) and from an own computer.
  * Google Drive: a full copy including the weights -> MyDrive/thesis_results/<name>/.
    Colab: through the mounted Drive. Own computer: through rclone (one-time setup in the README:
    a remote named config.RCLONE_REMOTE). Without rclone the weights stay in outputs/<model>/weights/.

`find_weights` gets the trained weights back (for LIME): from outputs/, Google Drive or rclone."""

import datetime
import os
import shutil
import subprocess
import sys
import zipfile

import torch

import config

IN_COLAB = "google.colab" in sys.modules


# ---------------------------------------------------------------- environment

def _git(*args, check=True, env=None):
    """Runs git in the repository and returns its output (stdout + stderr)."""
    r = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})})
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{out}")
    return out


def info():
    """Where and with which code a result was produced; stored in every summary.json."""
    try:
        commit = _git("rev-parse", "--short", "HEAD")
    except Exception:
        commit = "unknown"
    return {"machine": "colab" if IN_COLAB else "local",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
            "code_commit": commit,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}


def print_info():
    i = info()
    print(f"Running on: {i['machine']} | GPU: {i['gpu']} | code: commit {i['code_commit']}")
    if not torch.cuda.is_available():
        print("WARNING: no GPU found - training will be very slow.")


def ensure_dataset():
    """Downloads and unpacks the dataset into data/ if it is not there yet (Colab and own computer)."""
    if os.path.isdir(config.IMAGE_DIR):
        print(f"Dataset found: {config.IMAGE_DIR}")
        return
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    data_dir = os.path.join(config.ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "data.zip")
    print("Downloading the dataset (1.3 GB) ...")
    file_id = config.DATA_URL.split("/d/")[1].split("/")[0]     # id= works in old and new gdown versions
    gdown.download(id=file_id, output=zip_path, quiet=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    assert os.path.isdir(config.IMAGE_DIR), f"Dataset unpacked, but {config.IMAGE_DIR} is missing."
    print(f"Dataset ready: {config.IMAGE_DIR}")


# ---------------------------------------------------------------- saving the results

def _name(out_dir):
    """outputs/resnet50 -> "resnet50", outputs/lime/convnext_tiny -> "lime/convnext_tiny"."""
    return os.path.relpath(out_dir, config.OUTPUT_DIR).replace(os.sep, "/")


def save_results(out_dir, push=True):
    """Keeps the results in out_dir permanently (see the module docstring)."""
    name = _name(out_dir)
    if IN_COLAB:
        _save_to_drive(out_dir, name)
    else:
        save_to_drive_rclone(out_dir)
    _copy_to_repo(out_dir, name)
    if push:
        _push_to_github(name)


def _save_to_drive(out_dir, model):
    from google.colab import drive
    drive.mount("/content/drive")
    drive_dir = os.path.join(config.DRIVE_DIR, model)
    shutil.rmtree(drive_dir, ignore_errors=True)          # a re-run replaces this model's files only
    shutil.copytree(out_dir, drive_dir)
    drive.flush_and_unmount()
    print(f"Google Drive: full copy (with weights) in MyDrive/thesis_results/{model}/")


def _rclone():
    """Path of the rclone program, or None (also finds it in ~/.local/bin, which is often not on PATH)."""
    local = os.path.expanduser("~/.local/bin/rclone")
    return shutil.which("rclone") or (local if os.access(local, os.X_OK) else None)


def save_to_drive_rclone(out_dir):
    """Own computer: copies outputs/<model>/ (with weights) to MyDrive/thesis_results/<model>/ via rclone.
    A re-run replaces this model's files only. Can be called on its own to retry the upload."""
    model = _name(out_dir)
    remote = config.RCLONE_REMOTE
    setup = (f"  One-time setup: see the README, section 'Google Drive on an own computer'.\n"
             f"  Until then all outputs (with weights) are only on this computer: outputs/{model}/")
    exe = _rclone()
    if not exe:
        print(f"Google Drive: NOT saved - rclone is not installed.\n{setup}")
        return False
    remotes = subprocess.run([exe, "listremotes"], capture_output=True, text=True).stdout.split()
    if f"{remote}:" not in remotes:
        print(f"Google Drive: NOT saved - no rclone remote named '{remote}'.\n{setup}")
        return False
    dst = f"{remote}:thesis_results/{model}"
    print(f"Google Drive: uploading outputs/{model}/ ...")
    r = subprocess.run([exe, "sync", out_dir, dst], capture_output=True, text=True)
    if r.returncode:
        print(f"Google Drive: upload FAILED:\n{(r.stdout + r.stderr).strip()}\n"
              f"  Retry later in the notebook with:  env.save_to_drive_rclone(out_dir)")
        return False
    print(f"Google Drive: full copy (with weights) in MyDrive/thesis_results/{model}/")
    return True


def find_weights(model, seed):
    """Path of outputs/<model>/weights/<model>_seed<seed>.pth. If it is not on this computer, it is copied
    from Google Drive (MyDrive/thesis_results/<model>/weights/): in Colab through the mounted Drive,
    on an own computer through rclone."""
    name = f"{model}_seed{seed}.pth"
    local = os.path.join(config.OUTPUT_DIR, model, "weights", name)
    if os.path.isfile(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    print(f"{name} is not in outputs/{model}/weights/ - getting it from Google Drive ...")
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        src = os.path.join(config.DRIVE_DIR, model, "weights", name)
        if os.path.isfile(src):
            shutil.copy(src, local)
    elif _rclone():
        subprocess.run([_rclone(), "copyto", f"{config.RCLONE_REMOTE}:thesis_results/{model}/weights/{name}", local],
                       capture_output=True, text=True)
    if not os.path.isfile(local):
        raise FileNotFoundError(
            f"Weights not found: {name}.\n  Download MyDrive/thesis_results/{model}/weights/{name} from Google Drive "
            f"and put it into {local}\n  (or run this notebook where the model was trained).")
    return local


def _copy_to_repo(out_dir, model):
    """results/<model>/ = outputs/<model>/ without the weights."""
    dst = os.path.join(config.RESULTS_DIR, model)
    shutil.rmtree(dst, ignore_errors=True)
    for sub in ("figures", "results"):
        if os.path.isdir(os.path.join(out_dir, sub)):
            shutil.copytree(os.path.join(out_dir, sub), os.path.join(dst, sub))
    print(f"Copied figures and result files to results/{model}/")


def _github_token():
    """Colab only: the GITHUB_TOKEN secret (key icon in the left Colab sidebar), or None."""
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def _push_to_github(model):
    path = f"results/{model}"
    _git("add", "-A", "--", path)
    if not _git("status", "--porcelain", "--", path):
        print("GitHub: no changes to push.")
        return

    who = []
    if IN_COLAB or not _git("config", "user.email", check=False):
        who = ["-c", "user.name=CNN-LIME notebook", "-c", "user.email=notebook@users.noreply.github.com"]
    i = info()
    msg = f"Results: {model} ({i['machine']}, {i['gpu']}, code {i['code_commit']})"
    _git(*who, "commit", "-q", "-m", msg, "--", path)

    token = _github_token() if IN_COLAB else None
    if IN_COLAB and not token:
        print("GitHub: results committed in Colab but NOT pushed - no GITHUB_TOKEN secret found.\n"
              "  Add it once: Colab sidebar -> key icon (Secrets) -> name GITHUB_TOKEN, value = a GitHub\n"
              "  personal access token with 'Contents: read and write' for this repository; allow notebook\n"
              "  access, then run this cell again. (The results are safe on Google Drive either way.)")
        return
    try:
        _git(*who, "pull", "-q", "--rebase", "--autostash", "origin", "main")
        if token:
            url = _git("remote", "get-url", "origin").replace("https://", f"https://x-access-token:{token}@")
            _git("push", "-q", url, "HEAD:main")
        else:
            _git("push", "-q", "origin", "HEAD:main")
    except RuntimeError as e:
        text = str(e).replace(token, "***") if token else str(e)
        print(f"GitHub: committed locally but the push failed:\n{text}\n"
              "  Push later with:  git pull --rebase && git push")
        return
    print(f"GitHub: pushed {path}/ to the main branch.")
