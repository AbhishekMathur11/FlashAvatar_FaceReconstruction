# Setup and usage

This document covers (1) Docker setup and scripts, and (2) training and testing commands for FlashAvatar.

---

## 1. Docker setup

The project can be run in a Docker container with all dependencies pre-installed (no conda). The Docker-related files are stored in the **`docker_files/`** folder in this repo; the **working copies** that you actually run should live in the **SurfAvatars repo root** (the parent of `docker_ws/`), so that `docker_ws` exists next to the scripts.

### Files

| File | Purpose |
|------|--------|
| **Dockerfile** | Defines the image: CUDA 11.6, Python 3.8, venv, PyTorch 1.12, diff-gaussian-rasterization, simple-knn, and other pip deps. |
| **build.sh** | Builds the Docker image `surfavatars:latest`. |
| **run.sh** | Runs or attaches to the persistent container; mounts `docker_ws` so local edits are visible inside. |
| **terminal.sh** | Opens a new shell in the **same** running container. |

Copies of these files are in **`docker_files/`** in this directory. To use them, copy (or symlink) the four files into the **SurfAvatars root** (parent of `docker_ws/`), then run the scripts from there.

### How to use the Docker scripts

**1. Build the image (once)**  
From the **SurfAvatars repo root** (where the Dockerfile and `docker_ws/` live):

```bash
./build.sh
```

**2. Run the container (first time or after it was stopped)**  
From the same root:

```bash
./run.sh
```

- First run: creates a container named `surfavatars_dev` with `docker_ws` mounted at `/workspace`. Your shell is inside the container; you land in `/workspace/FlashAvatar_FaceReconstruction`.
- Exiting (e.g. `exit`) stops the container but keeps it; state (installed deps, files under `/root/persistent`) is preserved.
- Running `./run.sh` again starts the same container and attaches your terminal to it.

**3. Open another terminal in the same container**  
With the container already running (e.g. you left `run.sh` open in another terminal):

```bash
./terminal.sh
```

**4. GUI (e.g. Linux with X11)**  
If you need display inside the container, once per session on the host:

```bash
xhost +local:docker
```

### Behaviour summary

- **Persistent container:** Same container name (`surfavatars_dev`); restarting with `./run.sh` keeps the same state.
- **Live code:** `docker_ws` is bind-mounted to `/workspace`, so changes in `docker_ws` on the host are visible inside the container.
- **Working directory:** Inside the container you start in `/workspace/FlashAvatar_FaceReconstruction` (this project). All commands below are run from there.

---

## 2. Data layout

Expected layout (on host or inside container, relative to this project root):

```
dataset/
├── <idname>/
│   ├── alpha/      # raw alpha prediction
│   ├── imgs/       # extracted video frames
│   ├── parsing/    # semantic segmentation
│   └── <logname>/  # created by training
│       ├── train/  # training visualizations
│       └── ckpt/   # checkpoints (chkpnt5000.pth, chkpnt10000.pth, ...)
...
metrical-tracker/
└── output/
    └── <idname>/
        └── checkpoint/
```

Training reads from `dataset/<idname>/` and `metrical-tracker/output/<idname>/`, and writes to `dataset/<idname>/<logname>/`.

---

## 3. Training commands

Run from the project directory (e.g. inside the container: `/workspace/FlashAvatar_FaceReconstruction`).

**Default (identity `id1_25`, 150k iterations, log dir `log`):**
```bash
python train.py --idname id1_25
```

**Different identity:**
```bash
python train.py --idname your_id_name
```

**Different iteration count (e.g. 30k):**
```bash
python train.py --idname id1_25 --iterations 30000
```

**Different log name (separate run, different checkpoints):**
```bash
python train.py --idname id1_25 --logname log_30k
```

**Resume from a checkpoint:**
```bash
python train.py --idname id1_25 --logname log --start_checkpoint dataset/id1_25/log/ckpt/chkpnt10000.pth
```

**Change normal smoothness loss weight (default 0.01):**
```bash
python train.py --idname id1_25 --normal_loss_wt 0.02
```

**Different image resolution (default 512):**
```bash
python train.py --idname id1_25 --image_res 256
```

**Different random seed:**
```bash
python train.py --idname id1_25 --seed 42
```

**Combined example:**
```bash
python train.py --idname id1_25 --logname exp_30k --iterations 30000 --start_checkpoint dataset/id1_25/log/ckpt/chkpnt15000.pth --normal_loss_wt 0.02 --image_res 512 --seed 0
```

Checkpoints are saved every 5000 iterations as `dataset/<idname>/<logname>/ckpt/chkpnt<N>.pth`.

---

## 4. Testing / evaluation commands

**Basic (evaluate a trained checkpoint):**
```bash
python test.py --idname id1_25 --logname log --checkpoint dataset/id1_25/log/ckpt/chkpnt150000.pth
```

**Different identity and log:**
```bash
python test.py --idname your_id_name --logname log_30k --checkpoint dataset/your_id_name/log_30k/ckpt/chkpnt30000.pth
```

**Different image resolution:**
```bash
python test.py --idname id1_25 --logname log --checkpoint dataset/id1_25/log/ckpt/chkpnt150000.pth --image_res 256
```

**Different seed:**
```bash
python test.py --idname id1_25 --logname log --checkpoint dataset/id1_25/log/ckpt/chkpnt150000.pth --seed 42
```

The test script writes:
- **`dataset/<idname>/<logname>/test.avi`** – side-by-side GT vs rendered video  
- **`dataset/<idname>/<logname>/mesh.ply`** and **`mesh.obj`** – exported FLAME mesh  
- Prints per-frame and average **PSNR**, **L1**, **LPIPS**.

---

## 5. Quick reference

| Task | Command |
|------|--------|
| Train (default) | `python train.py --idname id1_25` |
| Train 30k iters | `python train.py --idname id1_25 --iterations 30000` |
| Train with new log name | `python train.py --idname id1_25 --logname my_run` |
| Resume training | `python train.py --idname id1_25 --start_checkpoint dataset/id1_25/log/ckpt/chkpnt10000.pth` |
| Test | `python test.py --idname id1_25 --logname log --checkpoint dataset/id1_25/log/ckpt/chkpnt150000.pth` |
| Normal loss weight | `python train.py --idname id1_25 --normal_loss_wt 0.02` |

---

## 6. Optional training arguments

From `arguments/__init__.py` you can also override optimization parameters, for example:

```bash
# Custom iterations and position learning rate
python train.py --idname id1_25 --iterations 50000 --position_lr_init 0.0002 --position_lr_final 0.000002

# Densification
python train.py --idname id1_25 --densify_from_iter 500 --densify_until_iter 15000 --densify_grad_threshold 0.0002
```

Default iterations is 150,000; checkpoints are saved every 5000.
