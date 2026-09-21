# We will release the data_loader part after the paper is accepted.


## Visualization Results 
![Visualization Results](images/pred_act1.gif)
![Visualization Results](images/pred_act2.gif)
![Visualization Results](images/pred_act5.gif)
## 🛠 Setup

### 1. Python environment

The CPU regression suite was verified on Windows with Python 3.12.14 and
PyTorch 2.14.0+cpu. The exact tested dependencies are in `requirements-test.txt`.
The old Python 3.8/PyTorch 1.7 instructions are not compatible with the current
versioned checkpoint API. `requirement.txt` is retained only as a historical list.

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

GPU training requires a PyTorch CUDA build suitable for the target machine; the
CPU suite does not validate GPU speed, memory use, or final prediction quality.

### 2. Datasets

Training/evaluation require the authors' private `data_loader/` sources and the
corresponding dataset. Set the dataset root with `--data_path` or its YAML setting.
HARPER uses 21 human and 21 retained Spot points. The authors confirmed that their
raw 23-point Spot arrays exclude camera-related points 21/22; both frame-rate
configurations retain indices 0..20. Loading and preprocessing apply this same
selection before centering, augmentation, or DCT. The model predicts 20 non-root
human joints and conditions on those 20 plus 21 Spot points. Other raw array sizes
are rejected to prevent silent index shifts. Upstream labels 21/22 as wrist/hand;
the [node audit](docs/harper_joint_definition_audit.md) distinguishes those labels
from the authors' terminology and input subset.
CHICO uses 15 human and 9 robot
joints, and the current CoMaD configuration is HR-only. 3DPW conditions on the
recorded second person. CMU human-human evaluation requires recorded synchronized
pairs; single-person TXT clips are not converted into artificial interactions.

## ⏳ To Training

Training on HARPER:

python main.py --cfg harper3d_30hz --mode train --exp_name harper3d_30hz

python main.py --cfg harper3d_120hz --mode train --exp_name harper3d_120hz

Training on CHICO:

python main.py --cfg chico --mode train --exp_name chico

Training on CoMad:

python main.py --cfg comad --mode train --exp_name comad


### 🔎 To Evaluation

Evaluate on HARPER:

python main.py --cfg harper3d_30hz --mode eval --ckpt ./results/harper3d_30hz/models/best_ema.pt

python main.py --cfg harper3d_120hz --mode eval --ckpt ./results/harper3d_120hz/models/best_ema.pt

Evaluate on CHICO:

python main.py --cfg chico --mode eval --ckpt ./results/chico/models/best_ema.pt

Evaluate on CoMad:

python main.py --cfg comad --mode eval --ckpt ./results/comad/models/best_ema.pt

## 🎥 To Visualization
Run the following scripts for visualization purpose:

python main.py --cfg harper3d_30hz --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/harper3d_30hz/models/best_ema.pt

python main.py --cfg harper3d_120hz --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/harper3d_120hz/models/best_ema.pt

Evaluate on CHICO:

python main.py --cfg chico --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/chico/models/best_ema.pt

Evaluate on CoMad:

python main.py --cfg comad --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/comad/models/best_ema.pt




## 🌹 Acknowledgment
Project structure is borrowed from [TransFusion](https://github.com/sibotian96/TransFusion). We would like to thank the authors for making their code publicly available.


