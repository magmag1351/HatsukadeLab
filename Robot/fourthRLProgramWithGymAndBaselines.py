import gymnasium as gym
import register  # これで SimpleGrid-v0 が使えるようになる

# 登録した環境を作成
env = gym.make("SimpleGrid-v0", size=5, render_mode="human")

# 環境をリセット
observation, info = env.reset()
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()  # ランダム行動
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")

env.close()
