# ConRFT-sim: Add MuJoCo Simulation to ConRFT

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://cccedric.github.io/conrft/)

This repository extends the original ConRFT with a MuJoCo-based Franka simulation environment (`franka_sim`), enabling full training, evaluation, and debugging in simulation before deployment to real robots.

## 🛠️ Installation Instructions (Updated & Verified)

The original installation process is no longer reliable due to severe version conflicts between CUDA, JAX, NumPy, Octo, and `serl_launcher`. The following workflow has been fully verified on a clean system. Please follow it exactly.

### 1. Environment and CUDA

```bash
conda create -n conrft python=3.10
conda activate conrft
conda install -c "nvidia/label/cuda-12.1.0" cuda
# This is not text error, to run this project, the quickest way to to create a new environment to satisfy jax and torch simutanously
conda create -n flexiv_conrft python=3.10
conda activate flexiv_conrft
conda install -c "nvidia/label/cuda-12.1.0" cuda
```

### 2. PyTorch and JAX (GPU)

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install --upgrade "jax[cuda12_local]==0.6.2" \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Configure runtime libraries (modify the Conda path if needed):

```bash
export LD_LIBRARY_PATH=/home/dx/miniconda3/envs/conrft/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/dx/miniconda3/envs/conrft/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

Verify:

```bash
python - << 'EOF'
import jax
print(jax.devices())
import torch
print(torch.cuda.is_available())
EOF
```

### 3. Clone ConRFT

```bash
git clone https://github.com/cccedric/conrft.git
cd conrft
```

### 4. Install Octo (patched)

```bash
git clone git@github.com:cccedric/octo.git
cd octo
pip install -e .
```

Edit `octo/requirements.txt` and remove \'jax == 0.4.20\'

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

### 5. Install serl\_launcher (patched)

```bash
cd serl_launcher
```

Edit `setup.py` and remove:

```text
opencv_python
```

Install and add OpenCV manually:

```bash
pip install -e .
pip install "opencv-python<=4.9.0.80"
```

Edit `serl_launcher/requirements.txt` and remove \'numpy, flax, tensorflow, pynput\'

Then install remaining dependencies:

```bash
conda install -c conda-forge pynput
pip install -r requirements.txt
cd ..
```

### 6. Download Octo model weights

```bash
export HF_ENDPOINT=https://huggingface.co
huggingface-cli download octo-models/octo-base-1.5 --local-dir ./octo-base-1.5
mv octo-base-1.5 octo_model
```

### 7. Install franka\_sim

```bash
cd franka_sim
pip install -e .
pip install -r requirements.txt
cd ..
```

If you are using NVIDIA RTX 5090 and encounter `ptxas too old`, try:

```bash
pip install nvidia-cuda-nvcc-cu12==12.9.86
```

### 8. Final checklist

* `jax.devices()` shows CUDA devices
* `torch.cuda.is_available()` returns `True`
* Octo is installed without forcing old JAX
* `serl_launcher` installs without OpenCV / NumPy conflicts
* `octo_model/` directory exists

You can now proceed with training or robot deployment.

For real Franka hardware setup and impedance controller configuration, see:

```
./serl_robot_infra/README.md
```

## 🚀 Quick Start (Simulation: pick\_cube\_sim)

This section demonstrates how to train a Franka arm in the MuJoCo simulation environment using HIL-SERL. The task `pick_cube_sim` requires the robot to grasp randomly appearing cubes and lift them by 0.1 m along the z-axis.

### 1. Enter the experiment directory

```bash
cd examples/experiments/pick_cube_sim/
```

### 2. Collect demonstration data

Example: collect 20 successful trajectories. Data will be saved to `examples/experiments/pick_cube_sim/demo_data`.

```bash
python ../../record_demos_octo.py --exp_name pick_cube_sim --successes_needed 20
```

### 3. Start human-in-the-loop RL training

Before running, edit `run_actor.sh` and `run_learner.sh`:

* `exp_name` must match a folder name under `experiments/`
* `demo_path` must be the **absolute path** to the `.pkl` demo file

Then start training:

```bash
bash run_actor_octo.sh --checkpoint_path first_run
bash run_learner_octo.sh
```

### 4. Evaluate the trained policy

Example configuration:

```bash
bash run_actor_octo.sh --eval_checkpoint_step=30000 --eval_n_trajs=100 --checkpoint_path=first_run
```

## Citation

```bibtex
@article{chen2025conrft,
  title={ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy},
  author={Chen, Yuhui and Tian, Shuai and Liu, Shugao and Zhou, Yingting and Li, Haoran and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2502.05450},
  year={2025}
}
```

```bibtex
@inproceedings{chen2025conrft,
  title={ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy},
  author={Yuhui Chen and Shuai Tian and Shugao Liu and Yingting Zhou and Haoran Li and Dongbin Zhao},
  booktitle={Proceedings of Robotics: Science and Syste
```


```
