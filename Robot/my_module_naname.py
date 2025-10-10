import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame  # for visualization

class SimpleGridEnv(gym.Env):
    """
    シンプルなグリッドワールド環境
    スタートとゴールを指定可能に変更（デフォルトはランダム）
    - 観測: エージェントと目標の位置
    - 行動: 上下左右斜め移動の8方向
    - 報酬: 目標に到達したら1.0、ステップごとに小さなコスト -0.01
 
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, render_mode=None, start=None, goal=None):
        self.size = size  # グリッドのサイズ
        self.window_size = 512  # 描画ウィンドウのサイズ
        self.render_mode = render_mode

        # 外部から指定された場合のスタート・ゴール
        self.fixed_start = start
        self.fixed_goal = goal

        # 観測空間: エージェントと目標の位置
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.float32),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.float32)
        })

        # 行動空間: 上下左右斜め移動の8方向
        self.action_space = spaces.Discrete(8)

        # 方向のマッピング
        self._action_to_direction = {
            0: np.array([1, 0]),    # 右
            1: np.array([0, 1]),    # 上
            2: np.array([-1, 0]),   # 左
            3: np.array([0, -1]),   # 下
            4: np.array([1, 1]),    # 右上    
            5: np.array([-1, 1]),   # 左上
            6: np.array([-1, -1]),  # 左下
            7: np.array([1, -1])    # 右下
        }

        # 描画用の変数
        self.window = None
        self.clock = None

        # 状態の初期化（resetで正式に初期化）
        self._agent_location = np.zeros(2, dtype=np.float32)
        self._target_location = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """環境を初期状態にリセットする"""
        super().reset(seed=seed)

        if self.fixed_start is not None:
            # 指定されたスタートの位置を使用
            self._agent_location = np.array(self.fixed_start, dtype=np.float32)
        else:
            # エージェントの位置をランダムに設定
            self._agent_location = self.np_random.integers(0, self.size, size=2).astype(np.float32)

        if self.fixed_goal is not None:
            # 指定されたゴールの位置を使用
            self._target_location = np.array(self.fixed_goal, dtype=np.float32)
        else:
            # 目標位置をエージェントと異なる位置に設定
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        # 描画モードが指定されていれば描画
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """行動を実行し、環境を1ステップ進める"""
        # 行動に対応する方向を取得
        direction = self._action_to_direction[int(action)]  # 安全のため int にキャスト

        # エージェントの位置を更新
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # 終了条件: エージェントが目標に到達
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = 1.0 if terminated else -0.01   #ゴールしたら報酬 1.0、ステップごとに小さなコスト -0.01

        truncated = False       # 時間制限による終了はなし
        observation = self._get_obs()
        info = self._get_info()

        # 描画モードが指定されていれば描画
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def _get_obs(self):
        """現在の観測を取得する"""
        return {
            "agent": self._agent_location.astype(np.float32),
            "target": self._target_location.astype(np.float32)
        }

    def _get_info(self):
        """追加情報を取得する（デバッグ用）"""
        return {
            "distance": float(np.linalg.norm(self._agent_location - self._target_location, ord=1))
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
        canvas.fill((255, 255, 255))  # 白背景

        # グリッドのマス目のサイズ（ピクセル）
        pix_square_size = self.window_size / self.size

        # 目標位置を赤い四角で描画
        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # 赤色
            pygame.Rect(
                tuple((pix_square_size * self._target_location).astype(int)),
                (int(pix_square_size), int(pix_square_size)),
            ),
        )

        # エージェントを青い円で描画
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # 青色
            tuple(((self._agent_location + 0.5) * pix_square_size).astype(int)),
            int(pix_square_size / 3),
        )

        # グリッド線を描画
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # 黒色
                (0, int(pix_square_size * x)),
                (self.window_size, int(pix_square_size * x)),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # 黒色
                (int(pix_square_size * x), 0),
                (int(pix_square_size * x), self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # 描画内容をウィンドウに反映
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # フレームレートを安定させる
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """環境のリソースを解放する"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
