"""Train LoDADA on one off-dynamics task.

Example:
    python train.py --config configs/mujoco_gravity_friction/halfcheetah-gravity-0.1.yaml --seed 0

Each task config (configs/<family>/<task>.yaml) holds the task definition and the
LoDADA hyperparameters used in the paper; it is merged on top of the domain-level
training config in configs/base/<domain>.yaml.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import d4rl  # noqa: F401  (registers the D4RL source-domain environments and datasets)
import gym
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from algo.buffers import LoDADAReplayBuffer, ReplayBuffer
from algo.lodada import LoDADA
from dataset.call_dataset import call_tar_dataset
from envs.adroit.call_adroit_env import call_adroit_env
from envs.antmaze.call_antmaze_env import call_antmaze_env
from envs.infos import get_normalized_score
from envs.mujoco.call_mujoco_env import call_mujoco_env

ROOT = Path(__file__).resolve().parent
CALL_ENV = {'mujoco': call_mujoco_env, 'adroit': call_adroit_env, 'antmaze': call_antmaze_env}


def load_config(config_path, params=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        task_cfg = yaml.safe_load(f)
    with open(ROOT / 'configs' / 'base' / f"{task_cfg['domain']}.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.update(task_cfg)
    if params:
        cfg.update(json.loads(params))
    cfg['task_name'] = Path(config_path).stem
    return cfg


def parse_shift_level(cfg):
    if cfg['domain'] == 'mujoco' and ('gravity' in cfg['env'] or 'friction' in cfg['env']):
        return float(cfg['shift_level'])
    return str(cfg['shift_level'])


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print(f"[{eval_cnt}] Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def set_seed(seed, envs):
    for env in envs:
        env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class Logger:
    """Writes evaluation results to TensorBoard, progress.csv and (optionally) W&B."""

    def __init__(self, outdir, cfg, use_wandb, wandb_project):
        self.outdir = outdir
        self.writer = SummaryWriter(str(outdir / 'tb'))
        self.csv_path = outdir / 'progress.csv'
        self.rows = []
        self.wandb = None
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=f"{cfg['task_name']}-seed{cfg['seed']}", config=cfg, dir=str(outdir))
            self.wandb = wandb

    def log(self, step, metrics):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, global_step=step)
        if self.wandb is not None:
            self.wandb.log({**metrics, 'step': step})
        self.rows.append({'step': step, **metrics})
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[-1].keys()))
            w.writeheader()
            w.writerows(self.rows)

    def close(self, final):
        with open(self.outdir / 'final.json', 'w') as f:
            json.dump(final, f, indent=2)
        self.writer.close()
        if self.wandb is not None:
            self.wandb.log({f'final/{k}': v for k, v in final.items()})
            self.wandb.finish()


def build_config(cfg, src_eval_env, device):
    cfg.update({
        'state_dim': src_eval_env.observation_space.shape[0],
        'action_dim': src_eval_env.action_space.shape[0],
        'max_action': float(src_eval_env.action_space.high[0]),
        'max_step': int(cfg['max_step']),
        'num_clusters': int(cfg['num_clusters']),
        'device': str(device),
    })
    return cfg


def run_standard(cfg, args, device):
    """Global dynamics shifts (MuJoCo gravity/friction/morphology), AntMaze map
    shifts and Adroit kinematic/morphology shifts with ODRL target datasets."""
    domain = cfg['domain']
    shift_level = parse_shift_level(cfg)
    ref_env_name = f"{cfg['env']}-{shift_level}"

    src_eval_env = gym.make(cfg['source_dataset'])
    src_eval_env.seed(args.seed)
    tar_eval_env = CALL_ENV[domain]({'env_name': cfg['env'], 'shift_level': shift_level})
    tar_eval_env.seed(args.seed + 100)

    outdir = make_outdir(args, cfg)
    logger = Logger(outdir, cfg, args.wandb, args.wandb_project)

    set_seed(args.seed, [src_eval_env, tar_eval_env])
    cfg = build_config(cfg, src_eval_env, device)
    save_config(outdir, cfg)

    policy = LoDADA(cfg, device)
    src_replay_buffer = LoDADAReplayBuffer(cfg, device)
    tar_replay_buffer = ReplayBuffer(device)

    tar_dataset = call_tar_dataset(cfg['env'], shift_level, cfg['tartype'])
    tar_replay_buffer.convert_D4RL(tar_dataset)
    if domain == 'antmaze':
        tar_replay_buffer.reward -= 1.0
    src_dataset = d4rl.qlearning_dataset(src_eval_env)
    src_replay_buffer.convert_D4RL(src_dataset, tar_dataset)
    if domain == 'antmaze':
        src_replay_buffer.reward -= 1.0

    if not cfg['noreg']:
        policy.train_vae(tar_replay_buffer, cfg['batch_size'], logger.writer)

    def evaluate(step, eval_cnt):
        src_ret = eval_policy(policy, src_eval_env, cfg['eval_episodes'], eval_cnt)
        tar_ret = eval_policy(policy, tar_eval_env, cfg['eval_episodes'], eval_cnt)
        return {'test/source return': src_ret, 'test/target return': tar_ret,
                'test/target normalized score': get_normalized_score(tar_ret, ref_env_name)}

    train_loop(cfg, policy, src_replay_buffer, tar_replay_buffer, evaluate, logger, args, outdir)


def run_localized(cfg, args, device):
    """Localized perturbations: target data are 5,000 transitions of the D4RL dataset;
    the source dataset is the D4RL dataset with region-wise Gaussian noise on states.
    The policy is evaluated in the unmodified environment."""
    src_eval_env = gym.make(cfg['source_dataset'])
    src_eval_env.seed(args.seed)

    outdir = make_outdir(args, cfg)
    logger = Logger(outdir, cfg, args.wandb, args.wandb_project)

    set_seed(args.seed, [src_eval_env])
    cfg = build_config(cfg, src_eval_env, device)
    save_config(outdir, cfg)

    policy = LoDADA(cfg, device)
    src_replay_buffer = LoDADAReplayBuffer(cfg, device)
    tar_replay_buffer = ReplayBuffer(device)

    # target data: a fixed random subset (seed 0) of the D4RL dataset
    tar_replay_buffer.convert_D4RL_local_tar(d4rl.qlearning_dataset(src_eval_env), num_samples=cfg['num_target_samples'])
    tar_dataset = {
        'observations': tar_replay_buffer.state,
        'actions': tar_replay_buffer.action,
        'next_observations': tar_replay_buffer.next_state,
        'rewards': tar_replay_buffer.reward,
        'terminals': 1. - tar_replay_buffer.not_done,
    }
    # source data: the full D4RL dataset with localized perturbations
    src_dataset = d4rl.qlearning_dataset(src_eval_env)
    src_replay_buffer.convert_D4RL_local(src_dataset, tar_dataset, cfg['local_shift_levels'], cfg['num_local_regions'])

    if not cfg['noreg']:
        policy.train_vae(tar_replay_buffer, cfg['batch_size'], logger.writer)

    def evaluate(step, eval_cnt):
        tar_ret = eval_policy(policy, src_eval_env, cfg['eval_episodes'], eval_cnt)
        # D4RL normalization of the unmodified environment ({env}-medium-v2 reference scores)
        return {'test/target return': tar_ret,
                'test/target normalized score': 100 * src_eval_env.get_normalized_score(tar_ret)}

    train_loop(cfg, policy, src_replay_buffer, tar_replay_buffer, evaluate, logger, args, outdir)


def train_loop(cfg, policy, src_replay_buffer, tar_replay_buffer, evaluate, logger, args, outdir):
    start = time.time()
    eval_cnt = 0
    metrics = evaluate(0, eval_cnt)
    eval_cnt += 1

    for t in range(cfg['max_step']):
        policy.train(src_replay_buffer, tar_replay_buffer, cfg['batch_size'], logger.writer)

        if (t + 1) % cfg['eval_freq'] == 0:
            metrics = evaluate(t + 1, eval_cnt)
            logger.log(t + 1, metrics)
            eval_cnt += 1
            if args.save_model:
                (outdir / 'models').mkdir(exist_ok=True)
                policy.save(str(outdir / 'models' / 'model'))

    final = {'step': cfg['max_step'], **metrics, 'runtime_sec': time.time() - start}
    print(f"Final step {cfg['max_step']}: " + ", ".join(f"{k} = {v:.2f}" for k, v in metrics.items()))
    logger.close(final)


def make_outdir(args, cfg):
    outdir = Path(args.dir) / cfg['task_name'] / f"seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_config(outdir, cfg):
    with open(outdir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='task config, e.g. configs/antmaze/antmaze-medium-play-1.yaml')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dir', default='./runs', help='output directory')
    parser.add_argument('--max_step', default=None, type=int, help='override the number of gradient steps')
    parser.add_argument('--params', default=None, help='JSON string to override config entries, e.g. \'{"num_clusters": 50}\'')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='also log to Weights & Biases')
    parser.add_argument('--wandb_project', default='LoDADA')
    args = parser.parse_args()

    cfg = load_config(args.config, args.params)
    if args.max_step is not None:
        cfg['max_step'] = args.max_step
    cfg['seed'] = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task: {cfg['task_name']} | seed: {args.seed} | device: {device}")

    if cfg.get('setting', 'standard') == 'localized':
        run_localized(cfg, args, device)
    else:
        run_standard(cfg, args, device)


if __name__ == '__main__':
    main()
