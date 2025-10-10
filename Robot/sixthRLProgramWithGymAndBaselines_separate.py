import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import register  # カスタム環境を登録（register.pyを読み込む）

# ===============================
# 学習（表示なし・並列環境）
# ===============================
n_envs = 1  # 並列実行する環境数
env_kwargs = {'size': 5}  # 環境に渡す引数（可視化しない）

# カスタム環境を並列で作成
train_env = make_vec_env("SimpleGrid-v0", n_envs=n_envs, env_kwargs=env_kwargs)

# PPOモデルを作成（MultiInputPolicyを使用）
model = PPO("MultiInputPolicy", train_env, verbose=1)

# 学習（ステップ数は適宜調整可能）
model.learn(total_timesteps=100000, progress_bar=True)

# 学習済みモデルを保存
model.save("ppo_vec_env_simplegrid")
del model  # メモリ解放（ロードテスト用）

# 学習環境を閉じる
train_env.close()

# ===============================
# 評価（表示あり・単一環境で可視化）
# ===============================
eval_env = gym.make("SimpleGrid-v0", size=5, render_mode="human")
model = PPO.load("ppo_vec_env_simplegrid")

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
