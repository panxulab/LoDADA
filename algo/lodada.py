"""LoDADA agent: IQL-style value learning on filtered source data plus target data,
with KL-based importance weights on source transitions and a CVAE regularizer
that keeps the policy close to the target behavior policy."""

import copy
import math
from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
from torch.optim.lr_scheduler import CosineAnnealingLR


class VAE_Policy(nn.Module):
    """CVAE that models the target-domain behavior policy."""

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim, device):
        super(VAE_Policy, self).__init__()
        if latent_dim is None:
            latent_dim = 2 * action_dim
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def dataset_prob(self, state: torch.Tensor, action: torch.Tensor, beta: float = 0.4, num_samples: int = 10) -> torch.Tensor:
        # num_samples is the number of sampled latent variables M in the paper
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta / 4)

        # p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1).logsumexp(dim=-1)
        return w

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_shared(torch.cat([state, action], -1))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)


class TanhTransform(Transform):
    r"""Transform via the mapping :math:`y = \tanh(x)`."""
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # No clamping here; use `cache_size=1` instead.
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # Numerically stable formula, see
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action * self.max_action, logprob, mean * self.max_action

    def bc_loss(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        pred_action = torch.tanh(mu) * self.max_action
        return torch.sum((pred_action - action) ** 2, dim=1)


class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class ValueFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class LoDADA(object):

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']

        # IQL hyperparameters
        self.lam = config['lam']    # expectile
        self.temp = config['temp']  # inverse temperature

        self.total_it = 0

        # alpha: importance weight of source samples in the critic loss, exp(-alpha * d_hat)
        # beta: BC weight on source samples, only used when the CVAE regularizer is disabled (noreg)
        self.alpha = config['alpha']
        self.beta = config['beta']

        # critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # value
        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        # actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, config['max_step'])

        # CVAE of the target behavior policy
        self.vae_policy = VAE_Policy(config['state_dim'], config['action_dim'], 2 * config['action_dim'], config['max_action'], 750, self.device).to(self.device)
        self.vae_policy_optimizer = torch.optim.Adam(self.vae_policy.parameters(), lr=1e-3)

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1, -1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def train_vae(self, tar_replay_buffer, batch_size, writer=None, num_steps=int(1e4)):
        """Pretrain the CVAE on target-domain (s, a) pairs."""
        for t in range(num_steps):
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
            recon, mean, std = self.vae_policy(tar_state, tar_action)
            recon_loss = F.mse_loss(recon, tar_action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            if t % 1000 == 0 and writer is not None:
                writer.add_scalar('train/VAE loss', vae_loss, t + 1)

            self.vae_policy_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_policy_optimizer.step()

    def update_target(self):
        """Moving-average update of the target critic."""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state_batch)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        v_loss = asymmetric_l2_loss(adv, self.lam)
        return v_loss, adv

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
        # source samples are weighted by exp(-alpha * d_hat); target samples have weight 1
        loss = (self.critic_weight * (q_1 - value_target) ** 2).mean() + (self.critic_weight * (q_2 - value_target) ** 2).mean()
        return loss

    def update_policy(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss_raw = self.policy.bc_loss(state_batch, action_batch)
        policy_loss = torch.mean(exp_adv * bc_loss_raw)

        if not self.config['noreg']:
            # CVAE regularizer: keep the policy close to the target behavior policy
            pi, _, _ = self.policy(state_batch)
            log_prob_behavior = self.vae_policy.dataset_prob(state_batch, pi, beta=0.4, num_samples=10)
            policy_loss -= self.config['reg_weight'] * log_prob_behavior.mean()
        elif self.beta != 0:
            bc_loss = torch.mean(self.bc_weight.squeeze(1) * bc_loss_raw)
            policy_loss += bc_loss
        return policy_loss

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1

        src_state, src_action, src_next_state, src_reward, src_not_done, src_divergence = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        # normalize the point-wise KL estimates within the batch
        src_divergence = (src_divergence - torch.max(src_divergence)) / (torch.max(src_divergence) - torch.min(src_divergence))

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)
        # target samples: critic weight 1, BC weight 0
        self.critic_weight = torch.ones_like(reward.flatten()).to(self.device)
        self.bc_weight = torch.zeros_like(reward.flatten()).to(self.device)

        self.critic_weight[:src_state.shape[0]] = torch.exp(-self.alpha * src_divergence).squeeze(-1)
        self.critic_weight = self.critic_weight.unsqueeze(1)
        self.bc_weight[:src_state.shape[0]] = torch.exp(-self.beta * src_divergence).squeeze(-1)
        self.bc_weight = self.bc_weight.unsqueeze(1)

        v_loss_step, adv = self.update_v_function(state, action, writer)
        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer)
        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step = self.update_policy(adv, state, action)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.policy_lr_schedule.state_dict(), filename + "_actor_lr_scheduler")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.v_func.load_state_dict(torch.load(filename + "_value"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.policy_lr_schedule.load_state_dict(torch.load(filename + "_actor_lr_scheduler"))
