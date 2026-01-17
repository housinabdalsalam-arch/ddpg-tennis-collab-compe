import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from maddpg_agent import MADDPG, MADDPGConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str)
    p.add_argument("--worker_id", type=int, default=1)
    p.add_argument("--no_graphics", action="store_true")
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--max_t", type=int, default=1000)
    p.add_argument("--prefix", type=str, default="checkpoint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = UnityEnvironment(file_name=args.env, worker_id=args.worker_id, no_graphics=args.no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print(f"[env] agents={num_agents}, state_size={state_size}, action_size={action_size}")

    cfg = MADDPGConfig()
    agent = MADDPG(state_size, action_size, num_agents=num_agents, seed=args.seed, cfg=cfg)

    scores_deque = deque(maxlen=100)
    scores_all = []

    best_avg = -1e9

    total_steps = 0

    for i_episode in range(1, args.episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()

        score_agents = np.zeros(num_agents, dtype=np.float32)

        for t in range(args.max_t):
            # warmup: pure random actions to kick-start sparse rewards
            if total_steps < cfg.warmup_steps:
                actions = np.random.uniform(-1.0, 1.0, size=(num_agents, action_size)).astype(np.float32)
            else:
                actions = agent.act(states, add_noise=True).astype(np.float32)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards, dtype=np.float32)
            dones = np.array(env_info.local_done, dtype=np.bool_)

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            score_agents += rewards
            total_steps += 1

            if np.any(dones):
                break

        agent.decay_noise()

        score = float(np.max(score_agents))  # Udacity scoring: max over both agents
        scores_deque.append(score)
        scores_all.append(score)

        avg = float(np.mean(scores_deque))

        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tScore: {score:.3f}\tAverage(100): {avg:.3f}\tNoise: {agent.noise_scale:.3f}")

        # save best rolling average (useful because training can be unstable)
        if avg > best_avg:
            best_avg = avg
            agent.save(prefix=args.prefix)

        # solved criterion
        if avg >= 0.50 and i_episode >= 100:
            print(f"\nSolved! Average(100)={avg:.3f} at episode {i_episode}")
            agent.save(prefix=args.prefix)
            break

    # plot
    plt.figure()
    plt.plot(np.arange(1, len(scores_all) + 1), scores_all)
    plt.ylabel("Score (max over agents)")
    plt.xlabel("Episode #")
    plt.title("Tennis (MADDPG) Training Scores")
    plt.grid(True)
    plt.savefig("scores.png", dpi=150)
    plt.close()

    env.close()


if __name__ == "__main__":
    main()
