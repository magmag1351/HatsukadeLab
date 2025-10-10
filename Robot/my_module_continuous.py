import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ContinuousGridEnv(gym.Env):
    """
    連続空間のグリッドワールド環境
    - エージェントは (x, y) 座標を連続的に移動
    - 行動は [dx, dy] の2次元連続ベクトル
    - 報酬:
        ゴールとの距離が小さくなれば報酬が増え(減点が減る)、
        ゴールに近づくと +1.0 の報酬を得て終了
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, size=5.0, render_mode=None, start=None, goal=None):
        super().__init__()

        self.size = size                # グリッドのサイズ
        self.window_size = 512          # 描画ウィンドウのサイズ
        self.render_mode = render_mode

        # 外部から指定された場合のスタート・ゴール
        self.fixed_start = start
        self.fixed_goal = goal

        # 状態と行動
        # 観測: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([size, size, size, size], dtype=np.float32),
            dtype=np.float32
        )

        # 行動: [dx, dy] の2次元ベクトル (-1～1 の範囲)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 状態の初期化（resetで正式に初期化）
        self._agent_location = np.zeros(2, dtype=np.float32)
        self._goal_location = np.zeros(2, dtype=np.float32)

        # 描画用
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """環境を初期状態にリセットする"""
        super().reset(seed=seed)

        if self.fixed_start is not None:
            # 指定されたスタートの位置を使用
            self._agent_location = np.array(self.fixed_start, dtype=np.float32)
        else:
            # エージェントの位置をランダムに設定
            self._agent_location = self.np_random.uniform(0, self.size, size=2).astype(np.float32)
        
        if self.fixed_goal is not None:
            # 指定されたゴールの位置を使用
            self._goal_location = np.array(self.fixed_goal, dtype=np.float32)
        else:
            # 目標位置をエージェントと異なる位置に設定
            self._goal_location = self.np_random.uniform(0, self.size, size=2).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        # 描画モードが指定されていれば描画
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """行動を実行し、環境を1ステップ進める"""
        # エージェント移動
        action = np.clip(action, -1, 1)
        self._agent_location += action * 0.1  # 移動スピード調整
        self._agent_location = np.clip(self._agent_location, 0, self.size)

        # 報酬
        distance = np.linalg.norm(self._agent_location - self._goal_location)
        
        reward = -distance * 0.05  # 距離が遠いほどマイナス
        
        terminated = distance < (self.size * 0.05)  # ゴール条件
        
        if terminated:
            reward = 1.0

        truncated = False       # 時間制限による終了はなし
        observation = self._get_obs()
        info = self._get_info()

        # 描画モードが指定されていれば描画
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """現在の観測を取得する"""
        return np.concatenate([self._agent_location, self._goal_location]).astype(np.float32)

    def _get_info(self):
        """追加情報を取得する（デバッグ用）"""
        return {
            "distance": float(np.linalg.norm(self._agent_location - self._goal_location))
        }

    def render(self):
        """環境を描画する"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """フレームを1枚描画する"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))    # 白背景

        # グリッドのマス目のサイズ（ピクセル）
        pix_square_size = self.window_size / self.size

        # ゴールを赤い四角で描画
        pygame.draw.circle(
            canvas,
            (255, 0, 0),    # 赤色
            (self._goal_location * pix_square_size).astype(int),
            10,
        )

        # エージェントを青い円で描画
        pygame.draw.circle(
            canvas,
            (0, 0, 255),    # 青色
            (self._agent_location * pix_square_size).astype(int),
            10,
        )

        if self.render_mode == "human":
            # 描画内容をウィンドウに反映
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # フレームレートを安定させる
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """環境のリソースを解放する"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
