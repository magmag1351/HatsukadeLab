import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import register  # カスタム環境登録（register.py）
import os

# ===============================
# TensorBoard用ログディレクトリ
# ===============================
log_dir = "./logs/simplegrid_ppo"
os.makedirs(log_dir, exist_ok=True)

# ===============================
# 設定
# ===============================
n_envs = 1  # 並列学習環境の数
env_kwargs = {'size': 5}  # グリッドサイズ指定

# ===============================
# 学習（表示なし・並列環境）
# ===============================
train_env = make_vec_env("SimpleGrid-v0", n_envs=n_envs, env_kwargs=env_kwargs)

# PPOモデル作成（tensorboard_log だけ指定）
model = PPO(
    "MultiInputPolicy",
    train_env,
    verbose=1,
    tensorboard_log=log_dir
)

# 学習実行
model.learn(total_timesteps=100000, progress_bar=True)

# モデル保存
model.save("ppo_simplegrid")
del model

train_env.close()

# ===============================
# 評価（表示あり・単一環境で可視化）
# ===============================
eval_env = gym.make(
    "SimpleGrid-v0",
    size=env_kwargs['size'],
    render_mode="human",
    start=(0, 0),
    goal=(4, 4)
)

model = PPO.load("ppo_simplegrid")

obs, info = eval_env.reset()
print(f"Starting observation: {obs}")
episode_over = False
total_reward = 0

while not episode_over:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
eval_env.close()
