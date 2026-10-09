"""
Play the trained Walker2d policy (v2_stand) in the MuJoCo viewer.

    python play.py                        # seed 10, speed 1.0 m/s
    python play.py --seed 12 --speed 0.5
    python play.py --speed 0              # standing command
"""
import argparse
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mywalker import make_env

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=10, help="training seed of the model: 10-14")
p.add_argument("--speed", type=float, default=1.0,
               help="commanded speed in m/s (0 = stand; trained range 0.5-1.5)")
p.add_argument("--episodes", type=int, default=1)
args = p.parse_args()

name = os.path.join(HERE, "models", f"walker_ppo_v2_stand_seed{args.seed}") 
kw = json.load(open(f"{name}_env_kwargs.json"))
kw["xml_file"] = os.path.join(HERE, os.path.basename(kw["xml_file"])) 
kw.update(speed_range=(args.speed, args.speed), stand_prob=0.0)        

model = PPO.load(name, device="cpu")
vecnorm = VecNormalize.load(f"{name}_vecnormalize.pkl", DummyVecEnv([lambda: make_env(**kw)]))
vecnorm.training, vecnorm.norm_reward = False, False      # use the saved observation statistics

env = make_env(render_mode="human", **kw)
for ep in range(args.episodes):
    obs, _ = env.reset(seed=200 + ep)
    done, ret, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        ret, steps, done = ret + r, steps + 1, terminated or truncated
    print(f"episode {ep}: {steps} steps, return {ret:.0f}, distance {info['x_position']:.2f} m, "
          f"{'fell' if terminated else 'did not fall'}")
env.close()
