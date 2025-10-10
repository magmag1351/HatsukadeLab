import gymnasium as gym
from stable_baselines3 import DQN
import register  # カスタム環境を登録する（register.pyを読み込む）
import os

# ===============================
# TensorBoard用ログディレクトリ
# ===============================
log_dir = "./logs/simplegrid_dqn"
os.makedirs(log_dir, exist_ok=True)

# ===============================
# 学習（表示なし）
# ===============================
train_env = gym.make("SimpleGrid-v0", size=5)  # render_mode指定なし（高速化）

model = DQN(
    policy="MultiInputPolicy",
    env=train_env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    tensorboard_log=log_dir  # ここを追加
)

# 学習ステップ数
model.learn(total_timesteps=100000, progress_bar=True)

# モデルを保存
model.save("dqn_simplegrid")
del model

train_env.close()

# ===============================
# 評価（表示あり）
# ===============================
eval_env = gym.make(
    "SimpleGrid-v0",
    size=5, 
    render_mode="human",
    start=(0, 0),
    goal=(4, 4)
)
model = DQN.load("dqn_simplegrid")

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
