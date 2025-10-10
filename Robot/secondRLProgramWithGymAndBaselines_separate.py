import gymnasium as gym
from stable_baselines3 import PPO

# ===============================
# 学習（表示なし）
# ===============================
train_env = gym.make("CartPole-v1")  # render_mode指定なし
model = PPO("MlpPolicy", train_env, verbose=1)

# 学習ステップ数（必要に応じて増やす）
model.learn(total_timesteps=100000)

# モデルを保存
model.save("ppo_cartpole")
del model  # 削除してロードをテスト

train_env.close()

# ===============================
# 評価（表示あり）
# ===============================
eval_env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole")

obs, info = eval_env.reset()
print(f"Starting observation: {obs}")
episode_over = False
total_reward = 0

while not episode_over:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
eval_env.close()
