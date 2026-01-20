# ConRFT-sim: Add mujoco simulation to ConRFT

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://cccedric.github.io/conrft/)

We provide examples to fine-tune Octo on top of HIL-SERL for robotic manipulation with human interventions.

## 🛠️ Installation Instructions (Updated & Verified)

> The original installation process is no longer reliable due to severe version conflicts between CUDA, JAX, NumPy, Octo and serl\_launcher. The following workflow has been fully verified on a clean system. Please follow it exactly.

### 1. Environment and CUDA

```bash
conda create -n flexiv_conrft python=3.10
conda activate flexiv_conrft
conda install -c "nvidia/label/cuda-12.1.0" cuda
```

### 2. PyTorch and JAX (GPU)

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install jax==0.4.26 jaxlib==0.4.26+cuda12.cudnn89 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Configure runtime libraries (modify the Conda path if needed):

```bash
export LD_LIBRARY_PATH=/home/dx/miniconda3/envs/flexiv_conrft/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/dx/miniconda3/envs/flexiv_conrft/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
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

Edit `octo/requirements.txt` and **remove or comment out**:

```text
jax == 0.4.20
```

Then install the remaining dependencies:

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

Install the package and OpenCV manually:

```bash
pip install -e .
pip install "opencv-python<=4.9.0.80"
```

Edit `serl_launcher/requirements.txt` and remove or comment out:

```text
numpy
flax
tensorflow
pynput
```

Then install the remaining dependencies:

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

### 7. Install franka_sim

```bash
cd franka_sim
pip install -e .
pip install -r requirements.txt & cd ..
```

if you use nvidia 5090 and find error "**ptxas too old**", then try "    pip install nvidia-cuda-nvcc-cu12==12.9.86    "

### 8. Final checklist

* `jax.devices()` shows CUDA devices
* `torch.cuda.is_available()` returns `True`
* Octo is installed without forcing old JAX
* serl\_launcher installs without OpenCV / NumPy conflicts
* `octo_model/` directory exists

You can now proceed with training or robot deployment.

### Real robot setup

For Franka robot and impedance controller configuration, see `./serl_robot_infra/README.md`.

## Contact

[chenyuhui2022@ia.ac.cn](mailto:chenyuhui2022@ia.ac.cn)

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
  booktitle={Proceedings of Robotics: Science and Systems (RSS) 2025}
}
```
