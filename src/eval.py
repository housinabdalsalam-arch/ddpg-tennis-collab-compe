import argparse
import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg_agent import MADDPG, MADDPGConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, type=str)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_t", type=int, default=1000)
    p.add_argument("--worker_id", type=int, default=1)
    p.add_argument("--no_graphics", action="store_true")

    # either use explicit actor paths...
    p.add_argument("--actor0", type=str, default=None)
    p.add_argument("--actor1", type=str, default=None)
    # ...or a prefix like "checkpoint" that loads checkpoint_actor_0/1.pth
    p.add_argument("--prefix", type=str, default="checkpoint")
    args = p.parse_args()

    env = UnityEnvironment(file_name=args.env, worker_id=args.worker_id, no_graphics=args.no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    agent = MADDPG(state_size, action_size, num_agents=num_agents, seed=0, cfg=MADDPGConfig())
    device = agent.device

    if args.actor0 is not None and args.actor1 is not None:
        agent.agents[0].actor_local.load_state_dict(torch.load(args.actor0, map_location=device))
        agent.agents[1].actor_local.load_state_dict(torch.load(args.actor1, map_location=device))
        agent.agents[0].actor_local.eval()
        agent.agents[1].actor_local.eval()
    else:
        agent.load_actors(prefix=args.prefix)

    scores = []
    for ep in range(1, args.episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        score_agents = np.zeros(num_agents, dtype=np.float32)

        for _ in range(args.max_t):
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = np.array(env_info.rewards, dtype=np.float32)
            dones = np.array(env_info.local_done, dtype=np.bool_)
            score_agents += rewards
            if np.any(dones):
                break

        score = float(np.max(score_agents))
        scores.append(score)
        print(f"Episode {ep:02d}\tScore: {score:.3f}")

    print("Average:", float(np.mean(scores)))
    env.close()


if __name__ == "__main__":
    main()
