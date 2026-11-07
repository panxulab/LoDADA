# LoDADA: Localized Dynamics-Aware Domain Adaptation for Off-Dynamics Offline Reinforcement Learning

### <p align="center">[NeurIPS 2026]</p>

<p align="center">
  Zhangjie Xia<sup>1,*</sup> ·
  <a href="https://yangyu0879.github.io/">Yu Yang</a><sup>2,*</sup> ·
  <a href="https://panxulab.github.io/">Pan Xu</a><sup>2</sup>
</p>
<p align="center">
<sup>1</sup> New York University ·
<sup>2</sup> Duke University ·
<sup>*</sup> Equal contribution
</p>

Code for the paper "Localized Dynamics-Aware Domain Adaptation for Off-Dynamics Offline Reinforcement Learning", Advances in Neural Information Processing Systems (NeurIPS) 2026.

## Installation

The code follows the [ODRL benchmark](https://github.com/OffDynamicsRL/off-dynamics-rl) and was tested with Python 3.8.13.

```bash
conda create -n lodada python=3.8.13 && conda activate lodada
pip install setuptools==63.2.0 wheel==0.38.4
pip install -r requirements.txt
```

`mujoco-py` needs the [MuJoCo 2.1.0 binaries](https://github.com/openai/mujoco-py#install-mujoco) in `~/.mujoco/mujoco210`.

## Datasets

- **Source domain.** D4RL datasets (`{halfcheetah,hopper,walker2d,ant}-medium-v2`, `antmaze-medium-{play,diverse}-v0`, `{pen,door}-human-v0`). D4RL downloads them to `~/.d4rl/datasets` on first use.
- **Target domain.** ODRL target datasets. Download them from the [ODRL dataset folder](https://drive.google.com/drive/folders/1fwkjtXCbMxVP7RM7NSN3mF40Gakwkqei?usp=sharing) and put the `.hdf5` files in `dataset/mujoco/`, `dataset/antmaze/` and `dataset/adroit/`.

## Experiments

Every task has one config file, `configs/<family>/<task>.yaml`, which sets the LoDADA hyperparameters used for that task in the paper. Shared training settings (network size, learning rates, IQL parameters, evaluation) are in `configs/base/<domain>.yaml`. The main LoDADA hyperparameters are:

- num_clusters: $K$ in the paper, number of clusters
- reg_weight: $\lambda$ in the paper, weight of the policy regularizer
- alpha: $\alpha$ in the paper, importance weight of source samples
- diameter_threshold: $\delta$ in the paper, distance threshold for source samples
- filter_ratio_small / filter_ratio_medium / filter_ratio_large: $(\xi_1,\xi_2,\xi_3)$ in the paper, filtering ratios
- sigma: $\sigma$ in the paper, input noise of the domain classifier
- max_step: number of gradient steps

To run one task with one seed:
```bash
python train.py --config configs/mujoco_gravity_friction/halfcheetah-gravity-0.1.yaml --seed 0
```
To run all tasks of a family with seeds 0–4 and summarize the final scores:
```bash
bash scripts/run.sh configs/mujoco_gravity_friction
python scripts/summarize.py --dir runs
```

### MuJoCo: gravity and friction shifts

HalfCheetah, Hopper, Walker2d and Ant with the gravity or friction coefficient scaled by 0.1, 0.5, 2.0 or 5.0. Source: D4RL medium; target: ODRL medium (5,000 transitions).

```bash
python train.py --config configs/mujoco_gravity_friction/ant-friction-2.0.yaml --seed 0
```

### MuJoCo: morphology shifts

Eight morphology shifts (e.g. `ant-morph-alllegs`, `hopper-morph-foot`) at the medium and hard levels.

```bash
python train.py --config configs/mujoco_morphology/hopper-morph-foot-hard.yaml --seed 0
```

### MuJoCo: localized perturbations

The target data are 5,000 transitions sampled from the D4RL medium dataset. The source dataset is the D4RL medium dataset split into 15 KMeans regions of the next state; the regions are divided into 3 or 5 groups, and Gaussian noise with the group's standard deviation (e.g. 0.1-0.5-2.0) is added to the states. The policy is evaluated in the unmodified environment.

```bash
python train.py --config configs/mujoco_localized/hopper-local-0.1-0.5-2.0.yaml --seed 0
```

### AntMaze

Six medium-size target maps (1–6) with the D4RL `antmaze-medium-play-v0` or `antmaze-medium-diverse-v0` source dataset.

```bash
python train.py --config configs/antmaze/antmaze-medium-diverse-3.yaml --seed 0
```

### Adroit

Pen and Door with kinematic (`broken-joint`) and morphology (`shrink-finger`) shifts at the medium and hard levels, with the D4RL human source dataset.

```bash
python train.py --config configs/adroit/door-broken-joint-medium.yaml --seed 0
```

## Citation
```
@article{xia2026localized,
  title={Localized Dynamics-Aware Domain Adaption for Off-Dynamics Offline Reinforcement Learning},
  author={Xia, Zhangjie and Yang, Yu and Xu, Pan},
  journal={arXiv preprint arXiv:2602.21072},
  year={2026}
}
```
