# Project 3 — Collaboration and Competition (Tennis)

This repo contains my solution for Udacity’s Deep Reinforcement Learning Nanodegree **Project 3: Collaboration and Competition**.

## Environment

Two agents play tennis with rackets. The goal is to keep the ball in play and score points.

- **Agents:** 2
- **Reward:**  
  - +0.1 if an agent hits the ball over the net and it lands in bounds  
  - -0.01 if an agent lets the ball hit the ground or hits out of bounds
- **State space:** 24 continuous values per agent (8 observations stacked × 3)
- **Action space:** 2 continuous actions per agent in **[-1, 1]**
- **Solved when:** average score **>= 0.5** over **100** consecutive episodes  
  (score per episode = **max** of the 2 agent scores)

## Approach — Multi-Agent DDPG (MADDPG)

I used **MADDPG**, which is a multi-agent extension of DDPG that enables **centralized training** with **decentralized execution**:

- Each agent has its own **Actor** policy:
  - `a_i = μ_i(s_i)`
- Each agent has a **centralized Critic** that conditions on the **joint** state and **joint** action:
  - `Q_i(s_1,...,s_N, a_1,...,a_N)`

This helps because the environment dynamics depend on both agents, and a centralized critic can learn a more stable value estimate.

### Update rule (high level)

From replay buffer samples:

**Target actions**
- For each agent:
  - `a'_i = μ_i_target(s'_i)`
- Joint target action:
  - `a' = [a'_1, ..., a'_N]`

**Critic target**
- Shared terminal condition: if **any** agent is done, treat the step as terminal.
- Target:
  - `y_i = r_i + γ * Q_i_target(S', A')`  (if not terminal)
  - `y_i = r_i` (if terminal)

**Critic loss**
- `L_critic = MSE(Q_i_local(S, A), y_i)`

**Actor loss**
- For agent `i`, optimize its action while keeping other agents fixed:
  - `L_actor = - mean( Q_i_local(S, [a_1,..., μ_i_local(s_i), ..., a_N]) )`

**Soft target updates**
- `θ_target = τ θ_local + (1-τ) θ_target`

## Stabilization components used

- **Replay buffer** for off-policy learning
- **Target networks** for both actor and critic
- **OU noise** with a decaying **noise scale** for exploration
- **Warmup random actions** (first few thousand steps) before learning begins
- **Gradient clipping** for critic updates

## Hyperparameters

From `MADDPGConfig` in `src/maddpg_agent.py` (main ones):

- **BUFFER_SIZE:** 1e6  
- **BATCH_SIZE:** 256  
- **GAMMA:** 0.99  
- **TAU:** 1e-3  
- **LR_ACTOR:** 1e-4  
- **LR_CRITIC:** 1e-3  
- **WEIGHT_DECAY:** 0.0  
- **UPDATE_EVERY:** 2 steps  
- **UPDATES_PER_STEP:** 2  
- **WARMUP_STEPS:** 5000  
- **OU noise:** theta = 0.15, sigma = 0.20  
- **Noise scale:** init = 1.0, min = 0.10, decay = 0.9995/episode  
- **Critic grad clip:** 1.0  

## Neural networks

Simple MLPs (Actor/Critic):

**Actor (per agent)**
- Input: 24  
- Hidden: 256 (ReLU)  
- Hidden: 256 (ReLU)  
- Output: 2 (Tanh)

Architecture: **24 → 256 → 256 → 2**

**Centralized Critic (per agent)**
- Input: full state = 2×24 = 48  
- Input: full action = 2×2 = 4  
- State path: 48 → 256 (ReLU)  
- Concatenate action (4)  
- 260 → 256 (ReLU) → 1

Architecture: **(48 → 256) + 4 → 256 → 1**

## Results

The environment is solved when the average score over 100 episodes is **>= 0.5**.

My agent solved the environment in **XXX episodes**, reaching an average score of **YY.YY** over the last 100 episodes.

![Training curve](scores.png)

Evaluation using the saved actor checkpoints achieved a mean score of **ZZ.ZZ** over 10 episodes.

## Future Work

- **Prioritized Experience Replay:** sample more informative transitions to speed up learning.
- **Parameter tuning:** batch size, update ratios, and noise schedule can improve stability.
- **Alternative algorithms:** PPO / SAC-style multi-agent variants can be explored for robustness.
