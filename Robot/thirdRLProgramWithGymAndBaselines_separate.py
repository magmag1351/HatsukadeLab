import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ===============================
# 学習（表示なし、並列9環境）
# ===============================
n_envs = 9
train_env = make_vec_env("CartPole-v1", n_envs=n_envs)  # 並列学習環境

model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=250000)  # 学習ステップ数は必要に応じて増やす

# モデルを保存
model.save("ppo_cartpole_vec_env")
del model  # 削除してロードをテスト

train_env.close()

# ===============================
# 評価（表示あり、単一環境で可視化）
# ===============================
eval_env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole_vec_env")

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
