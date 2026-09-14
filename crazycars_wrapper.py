import gymnasium as gym
import numpy as np

class CrazyCarsRewardWrapper(gym.Wrapper):
    """
    Reward shaping for the 'Crazy Cars' microgame.
    Reward shaping is based on blue-vs-green hue balance on the boombox screen.
    """

    def __init__(self, env, debug=False):
        super().__init__(env)
        self.prev_obs = None
        self.frame_count = 0
        self.last_event = None
        self.debug = debug

        # Screen regions
        h, w = env.observation_space.shape[:2]
        self.boombox_region = (slice(15, 55), slice(35, 65))
        self.play_region = (slice(10, 90), slice(8, 92))
        self.right_region_x = slice(int(w * 0.60), w)

        self.in_microgame = False
        self.phase_grace = 0
        self.end_buffer = []
        self.exit_pending = 0

    # Helper Functions
    def _region_mean(self, obs_f, region):
        y, x = region
        return float(np.mean(obs_f[y, x]))

    def _boombox_brightness(self, obs_f):
        return self._region_mean(obs_f, self.boombox_region)

    def _field_dark_ratio(self, obs_f):
        y, x = self.play_region
        pf = obs_f[y, x]
        return float(np.mean(pf < 35))

    def _is_microgame(self, obs_f):
        dark_ratio = self._field_dark_ratio(obs_f)
        bb_mean = self._boombox_brightness(obs_f)
        top_band = obs_f[0:25, :]
        top_mean = float(np.mean(top_band))
        if top_mean > 100 and bb_mean > 60:
            return False
        return (dark_ratio > 0.65) and (bb_mean < 75)

    def _boombox_face_color(self, obs_f):
        face_region = (slice(11, 38), slice(37, 62))
        y, x = face_region
        region = obs_f[y, x]
        if region.ndim == 2:
            v = float(np.mean(region))
            return v, v, v
        elif region.ndim == 3 and region.shape[-1] >= 3:
            r = float(np.mean(region[..., 0]))
            g = float(np.mean(region[..., 1]))
            b = float(np.mean(region[..., 2]))
            return r, g, b
        else:
            v = float(np.mean(region))
            return v, v, v

    # Gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.frame_count = 0
        self.in_microgame = self._is_microgame(obs.astype(np.float32))
        self.phase_grace = 12
        self.end_buffer.clear()
        self.exit_pending = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        obs_f = obs.astype(np.float32)
        total_reward = 0.05

        now_in_microgame = self._is_microgame(obs_f)

        # Transition Detection
        if now_in_microgame != self.in_microgame:
            if now_in_microgame:
                if self.debug:
                    print(f"[ENTER MICROGAME] Frame {self.frame_count}")
            else:
                self.exit_pending = 45
                self.end_buffer = []
                if self.debug:
                    print(f"[DEBUG] Exited microgame at frame {self.frame_count}, waiting 45 frames for evaluation.")
            self.in_microgame = now_in_microgame
            self.phase_grace = 8
            self.prev_obs = obs
            return obs, total_reward, terminated, truncated, info

        if self.phase_grace > 0:
            self.phase_grace -= 1
            self.prev_obs = obs
            return obs, total_reward, terminated, truncated, info

        # Post-microgame face evaluation
        if self.exit_pending > 0:
            self.exit_pending -= 1
            self.end_buffer.append(obs_f.copy())

            if self.exit_pending == 0 and len(self.end_buffer) > 0:
                avg_face = np.mean(self.end_buffer[-10:], axis=0)
                r, g, b = self._boombox_face_color(avg_face)
                total = r + g + b + 1e-5
                blue_ratio = b / total
                green_ratio = g / total
                brightness = total / 3

                # Final tuned classification
                if brightness > 80 and blue_ratio >= 0.34 and (blue_ratio - green_ratio) >= -0.04:
                    if self.debug:
                        print(f"[WIN] Frame {self.frame_count}: Wario smiling (blue/cyan face)")
                    total_reward += 10.0
                else:
                    if self.debug:
                        print(f"[LOSE] Frame {self.frame_count}: Wario frowning (green/gray face)")
                    total_reward -= 15.0

                self.end_buffer.clear()

        # Continuous microgame reward
        if self.in_microgame:
            total_reward += 0.02

        # Encourage jump attempts
        if isinstance(action, np.ndarray):
            if action.size == 1:
                act = int(action.item())
            else:
                act = int(action.flat[0])  # take first element safely
        elif isinstance(action, (list, tuple)):
            act = int(action[0])
        else:
            act = action

        if self.in_microgame and act in (1, 6, 7):
            total_reward += 1

        self.prev_obs = obs
        return obs, total_reward, terminated, truncated, info
