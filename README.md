# Walker2d: gait-style rewards with speed commands

A Walker2d policy that follows speed commands (0, 0.5–1.5 m/s) with a symmetric, trained with PPO.
```bash
pip install -r requirements.txt
python play.py                        # seed 10, 1.0 m/s
python play.py --seed 12 --speed 0.5  # other seed / speed
python play.py --speed 0              # standing command
```

`play.py`
Models for 5 training seeds (10–14) are included. The trained speed range is 0.5–1.5 m/s, plus 0 (stand).

## Files

| File | Content |
|---|---|
| `mywalker.py` | Environment and reward |
| `walker2d_hipdmc.xml` | Gymnasium Walker2d-v5 model with the hip range changed to [−20°,100°] (DeepMind Control Suite walker) |
| `play.py` | Loads a trained policy and runs it in the MuJoCo viewer |
| `models/walker_ppo_v2_stand_seed*.zip` | Trained PPO policies |
| `models/*_vecnormalize.pkl` | Observation normalization statistics |
| `models/*_env_kwargs.json` | Environment settings used in training |
| `notebooks/my_walker2d.ipynb` | Training and evaluation code |

## Setup

- **Environment:** Gymnasium Walker2d-v5 (MuJoCo), dt = 0.008 s, 1000 steps per episode.
  The default hip range [−150°, 0°] does not let the thigh swing forward relative to the torso,
  so it was replaced by the DeepMind Control Suite range [−20°,100°].
- **Command:** a target speed v\* sampled per episode from 0.5–1.5 m/s and appended to the observation;
  20% of the episodes use v\* = 0 (standing).
- **Algorithm:** PPO (Stable-Baselines3) with RL Zoo Walker2d hyperparameters, 8 parallel envs,
  2M steps, 5 seeds.
- **Curricula:** style terms phased in between 0.2M and 0.6M steps;
  speed range widened from 0.9–1.1 to 0.5–1.5 m/s between 0.4M and 1M steps.

## Reward

r = TRACK + HEALTHY + SYM + s(t)·(HEIGHT − FLIGHT − STEP − STAND) − CTRL, clipped at 0.

| Term | Definition |
|---|---|
| TRACK | 2 · exp(−((v − v\*)/0.25)²) |
| HEALTHY | +1 while the torso height and pitch are in range |
| SYM | 1 · ρ · exp(−ẽ): right leg now vs left leg half a step cycle earlier; ẽ normalized by the joint-angle variance, ρ = left/right amplitude ratio |
| HEIGHT | at each touchdown: 70 · τ · min(h / 0.08 m, 1), h = peak foot clearance, τ = step time |
| FLIGHT | −2 when both feet carry < 10% of the body weight |
| STEP | at each touchdown: up to −20 if the step is shorter than 0.3·√v\* m; −20 if the same foot lands twice |
| STAND | when v\* = 0: gait terms off, −1 · mean \|q − q_upright\| |
| CTRL | 0.001 · Σa² |

The half step cycle used by SYM is measured from foot touchdowns.

## Results
Gait quality varies across seeds: seed 10 is close to symmetric.
