import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


class MADDPGConfig:
    # Replay
    buffer_size = int(1e6)
    batch_size = 256

    # Discount / target update
    gamma = 0.99
    tau = 1e-3

    # Learning rates
    lr_actor = 1e-4
    lr_critic = 1e-3
    weight_decay = 0.0


    learn_every = 1
    learn_updates = 2
    warmup_steps = 5000  


    noise_mu = 0.0
    noise_theta = 0.15
    noise_sigma = 0.20
    noise_scale_start = 1.0
    noise_scale_decay = 0.9995
    noise_scale_min = 0.10

    critic_grad_clip = 1.0


class OUNoise:
    def __init__(self, size, seed=0, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.randn(*self.state.shape)
        self.state = self.state + dx
        return self.state


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0, device="cpu"):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        random.seed(seed)

        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"],
        )

    def add(self, states, actions, rewards, next_states, dones):

        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.states for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.actions for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.stack([e.rewards for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_states for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.stack([e.dones for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, seed, cfg, device):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        self.cfg = cfg
        self.device = device

        full_state_size = state_size * num_agents
        full_action_size = action_size * num_agents

        self.actor_local = Actor(state_size, action_size, seed=seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed=seed).to(device)

        self.critic_local = Critic(full_state_size, full_action_size, seed=seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, seed=seed).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=cfg.lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay
        )

        self.noise = OUNoise(
            size=action_size,
            seed=seed,
            mu=cfg.noise_mu,
            theta=cfg.noise_theta,
            sigma=cfg.noise_sigma,
        )

        self._hard_update(self.actor_target, self.actor_local)
        self._hard_update(self.critic_target, self.critic_local)

    def reset(self):
        self.noise.reset()

    def act(self, state, noise_scale=0.0):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().numpy().squeeze(0)
        self.actor_local.train()

        if noise_scale > 0.0:
            action = action + noise_scale * self.noise.sample()

        return np.clip(action, -1.0, 1.0)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)


class MADDPG:
    def __init__(self, state_size, action_size, num_agents=2, seed=0, cfg=None):
        self.cfg = cfg if cfg is not None else MADDPGConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.agents = [
            DDPGAgent(state_size, action_size, num_agents, seed + i, self.cfg, self.device)
            for i in range(num_agents)
        ]

        self.memory = ReplayBuffer(self.cfg.buffer_size, self.cfg.batch_size, seed=seed, device=self.device)
        self.t_step = 0

        self.noise_scale = self.cfg.noise_scale_start

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, states, add_noise=True):
        # states: (N, S)
        actions = []
        for i, agent in enumerate(self.agents):
            ns = self.noise_scale if add_noise else 0.0
            actions.append(agent.act(states[i], noise_scale=ns))
        return np.vstack(actions)

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        self.t_step += 1
        if len(self.memory) < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return

        if self.t_step % self.cfg.learn_every != 0:
            return

        for _ in range(self.cfg.learn_updates):
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences


        B = states.shape[0]
        full_states = states.reshape(B, -1)
        full_actions = actions.reshape(B, -1)
        full_next_states = next_states.reshape(B, -1)

        # Precompute target actions for all agents (no grad)
        with torch.no_grad():
            next_actions_all = []
            for j, ag in enumerate(self.agents):
                a_j = ag.actor_target(next_states[:, j, :])
                next_actions_all.append(a_j)
            full_next_actions = torch.cat(next_actions_all, dim=1)

        for i, agent in enumerate(self.agents):
            r_i = rewards[:, i].unsqueeze(1)
            d_i = dones[:, i].unsqueeze(1)

            # -------- critic --------
            with torch.no_grad():
                q_next = agent.critic_target(full_next_states, full_next_actions)
                q_target = r_i + self.cfg.gamma * (1.0 - d_i) * q_next

            q_expected = agent.critic_local(full_states, full_actions)
            critic_loss = F.mse_loss(q_expected, q_target)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.cfg.critic_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), self.cfg.critic_grad_clip)
            agent.critic_optimizer.step()

            # -------- actor --------

            actions_pred_all = []
            for j, ag_j in enumerate(self.agents):
                a_j = ag_j.actor_local(states[:, j, :])
                if j != i:
                    a_j = a_j.detach()
                actions_pred_all.append(a_j)
            full_actions_pred = torch.cat(actions_pred_all, dim=1)

            actor_loss = -agent.critic_local(full_states, full_actions_pred).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()

            agent.soft_update(agent.critic_local, agent.critic_target, self.cfg.tau)
            agent.soft_update(agent.actor_local, agent.actor_target, self.cfg.tau)

    def decay_noise(self):
        self.noise_scale = max(self.cfg.noise_scale_min, self.noise_scale * self.cfg.noise_scale_decay)

    def save(self, prefix="checkpoint"):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"{prefix}_actor_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"{prefix}_critic_{i}.pth")

    def load_actors(self, prefix="checkpoint"):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load(f"{prefix}_actor_{i}.pth", map_location=self.device))
            agent.actor_local.eval()
