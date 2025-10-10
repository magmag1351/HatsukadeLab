import gymnasium as gym
from stable_baselines3 import PPO
import register  # カスタム環境を登録する（register.pyを読み込む）

# ===============================
# 学習（トレーニング）
# ===============================
train_env = gym.make("ContinuousGrid-v0", size=5)  # render_mode指定なし
model = PPO("MlpPolicy", train_env, verbose=1)

# 学習ステップ数（調整可能）
model.learn(total_timesteps=200000, progress_bar=True)

# モデルを保存
model.save("ppo_continuous_grid")
del model  # テストのために削除

train_env.close()

# ===============================
# 評価（表示あり）
# ===============================
eval_env = gym.make(
    "ContinuousGrid-v0",
    size=5,
    render_mode="human",
    start=(0.5, 0.5),
    goal=(4.5, 2.5)
)

model = PPO.load("ppo_continuous_grid")

obs, info = eval_env.reset()
print(f"Starting observation: {obs}")
episode_over = False
total_reward = 0

while not episode_over:
    # PPOの予測（確率的ポリシーから連続値の行動を出す）
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
eval_env.close()
