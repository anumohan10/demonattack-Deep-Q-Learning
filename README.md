# LLM Agents & Deep Q-Learning with Atari Games

**Author**: Anusree Mohanan
**Environment**: `DemonAttack-v5`
**Framework**: Stable-Baselines3 (SB3) + SB3-Contrib
**License**: MIT

---

## 🚀 Overview

This project implements and compares Deep Reinforcement Learning agents (DQN, ε-greedy DQN, and QRDQN with NoisyNet) on the Atari environment **DemonAttack-v5**. Using Stable-Baselines3 and TensorBoard, the agents are trained, evaluated, and visualized across 100,000 timesteps.

---

## 📦 Installations

```bash
%pip install gym
%pip install opencv-python
%pip install "gymnasium[atari,accept-rom-license]"
%pip install tensorboard
%pip install moviepy
%pip install sb3-contrib
```

To fix version mismatches:

```bash
%pip uninstall gymnasium ale-py -y
%pip install "gymnasium[atari]==0.29.1" ale-py==0.8.1
```

---

## 🧠 Environment Setup

```python
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env("ALE/DemonAttack-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
```

**Environment Details:**

* Action Space: `Discrete(6)`
* Observation Shape: `(84, 84, 4)` (stacked grayscale frames)

---

## 🧪 Baseline DQN Training

```python
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.0001,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./dqn_demonattack_tensorboard/"
)
model.learn(total_timesteps=100_000, tb_log_name="dqn_cleanrun")
model.save("dqn_demonattack")
```

**Evaluation:**

```python
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

> **Result**: 426.25 ± 293.94

---

## 🎮 Video Recording

```python
from gymnasium.wrappers import RecordVideo

env = RecordVideo(env, video_folder="./videos", name_prefix="dqn_demo")
```

> Records 3 episodes of trained DQN gameplay.

---

## 🧪 ε-Greedy DQN Training

```python
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.0005,
    gamma=0.90,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./dqn_demonattack_tensorboard/"
)
model.learn(total_timesteps=100_000, tb_log_name="dqn_epsilon")
model.save("dqn_epsilon")
```

> **Result**: 220.75 ± 78.54 (20 episodes)

---

## 📊 Performance Summary

### Learning Rates

| Model            | Learning Rate |
| ---------------- | ------------- |
| dqn\_cleanrun\_1 | 0.0001        |
| dqn\_epsilon\_1  | 0.0005        |

### Exploration Strategy

| Model            | Strategy        |
| ---------------- | --------------- |
| dqn\_cleanrun\_1 | Default         |
| dqn\_epsilon\_1  | Custom ε-greedy |

---

## 📈 TensorBoard Highlights (DQN)

* `ep_len_mean`: Gradual rise from 700 to 1060
* `ep_rew_mean`: Improved from 90 to 260
* `exploration_rate`: Dropped to 0.05 early
* `fps`: \~90 (MacBook Air)
* `loss`: Low, stable (0.014–0.03)

---

## 🔬 Q-Value Inspection

```python
obs_tensor = torch.tensor(obs).float().permute(0, 3, 1, 2).to(model.device)
q_values = model.q_net(obs_tensor).cpu().numpy()
print("Q-values:", q_values)
print("Action:", np.argmax(q_values))
```

> Reveals model's internal confidence across actions.

---

## ⚙ QRDQN with NoisyNet

```python
from sb3_contrib import QRDQN
model_noisy = QRDQN(
    "CnnPolicy",
    env,
    learning_rate=0.0005,
    gamma=0.99,
    exploration_initial_eps=0.0,
    exploration_final_eps=0.0,
    verbose=1,
    tensorboard_log="./dqn_demonattack_tensorboard/"
)
model_noisy.learn(total_timesteps=100_000, tb_log_name="qrdqn_noisynet")
model_noisy.save("qrdqn_noisynet")
```

**Why NoisyNet?**

> Removes need for ε decay. Injects trainable noise for exploration.

**Evaluation Result**:

```text
Mean reward: 423.75 ± 288.37
```

### TensorBoard Stats

* `ep_rew_mean`: \~327
* `ep_len_mean`: \~1013
* `loss`: Higher (\~4.5) due to quantile regression
* `fps`: \~73

---

## 🏁 Conclusion

* **Baseline DQN**: Good learning, stable rewards
* **ε-Greedy Variant**: Higher learning rate, more exploration, improved reward
* **QRDQN + NoisyNet**: State-aware exploration, less hyperparameter tuning

---

## 📝 License

MIT License © 2025 Anusree Mohanan
