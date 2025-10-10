from gymnasium.envs.registration import register

# 環境の登録：my_module.py
register(
    id="SimpleGrid-v0",
    entry_point="my_module:SimpleGridEnv",  # my_module.py の SimpleGridEnv を使う
    max_episode_steps=10000,  # エピソードの最大ステップ数
)

# 環境の登録：my_module_naname.py
register(
    id="SimpleGrid-v1",
    entry_point="my_module_naname:SimpleGridEnv",  # my_module_naname.py の SimpleGridEnv を使う
    max_episode_steps=10000,  # エピソードの最大ステップ数
)

# 環境の登録：my_module_continuous.py
register(
    id="ContinuousGrid-v0",
    entry_point="my_module_continuous:ContinuousGridEnv",  # my_module_continuous.py の ContinuousGridEnv を使う
    max_episode_steps=10000,  # エピソードの最大ステップ数
)
