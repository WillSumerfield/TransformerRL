"""Random-action agent on Ant-v5. Sanity check for MuJoCo + gymnasium."""
import argparse
import gymnasium as gym


def run(episodes: int, seed: int, env_id: str, render: bool) -> None:
    env = gym.make(env_id, render_mode="human" if render else None)
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        ret, steps, done = 0.0, 0, False
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"ep {ep}: return={ret:.2f}  steps={steps}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, default="Ant-v5", dest="env_id")
    p.add_argument("--no-render", action="store_false", dest="render")
    run(**vars(p.parse_args()))
