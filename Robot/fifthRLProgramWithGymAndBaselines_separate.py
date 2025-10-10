import gymnasium as gym
from stable_baselines3 import PPO
import register  # カスタム環境を登録する（register.pyを読み込む）

# ===============================
# 学習（表示なし）
# ===============================
train_env = gym.make("SimpleGrid-v1", size=5)  # render_mode指定なし
model = PPO("MultiInputPolicy", train_env, verbose=1)

# 学習ステップ数（必要に応じて調整）
model.learn(total_timesteps=100000, progress_bar=True)

# モデルを保存
model.save("ppo_simplegrid")
del model  # 削除してロードをテスト

train_env.close()

# ===============================
# 評価（表示あり）
# ===============================
eval_env = gym.make(
    "SimpleGrid-v1",
    size=5,
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
