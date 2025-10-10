import gymnasium as gym
from stable_baselines3 import PPO

# 環境を可視化ありで作成
env = gym.make("CartPole-v1", render_mode="human")

# PPO モデルを作成
model = PPO("MlpPolicy", env, verbose=1)

# 学習（10000 ステップ）
model.learn(total_timesteps=10000)

# モデルを保存
model.save("ppo_cartpole")
del model  # 削除してロードをテスト

# モデルをロード
model = PPO.load("ppo_cartpole")

# 学習済みモデルで動かす
obs, info = env.reset()
print(f"Starting observation: {obs}")
episode_over = False
total_reward = 0
while not episode_over:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
