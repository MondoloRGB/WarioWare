import cv2
import numpy as np
import gymnasium as gym
from collections import deque


class MugShotRewardWrapper(gym.Wrapper):
    """
    Reward shaping for the 'Mug Shot' microgame.

    Detects when gameplay starts (mug appears), tracks the mug's position,
    and gives reward shaping for proximity to Wario's hand, pressing A, and win/loss.
    """

    def __init__(self, env, debug=False):
        super().__init__(env)
        self.debug = debug

        self.BUTTON_A = 8
        self.has_tried_catch = False
        self.frame_count = 0
        self.win_detected = False
        self.lose_detected = False
        self.game_active = False
        self.color_window = deque(maxlen=5)
        self.yellow_thresh = 160
        self.black_thresh = 40
        self.absent_streak = 0
        self.freeze_timer = 0

        # Region of interest for color sampling (exclude table + Wario)
        h, w = env.observation_space.shape[:2]
        self.play_region = (slice(int(h * 0.10), int(h * 0.45)), slice(int(w * 0.1), int(w * 0.45)))

        # Visualization color cache
        self.last_vis = None

    # Helper Functions
    def _region_rgb_mean(self, obs, region):
        y, x = region
        reg = obs[y, x]
        return np.mean(reg[..., 0]), np.mean(reg[..., 1]), np.mean(reg[..., 2])

    def _find_mug(self, obs):
        """
        Detects the mug in the current frame and computes its position and proximity to Wario's hand zones.
        """
        # Convert to BGR for OpenCV processing
        bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape

        # Mug color mask: gray-white neutral region
        lower = np.array([170, 170, 170], dtype=np.uint8)
        upper = np.array([235, 235, 235], dtype=np.uint8)
        mask = cv2.inRange(bgr, lower, upper)

        # Neutral filter (avoid colored backgrounds)
        b, g, r = cv2.split(bgr)
        diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
        diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))
        diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
        neutral_mask = ((diff_rg < 25) & (diff_rb < 25) & (diff_gb < 25)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, neutral_mask)

        # Region of Interest (where the mug typically moves)
        roi_y1, roi_y2 = int(h * 0.52), int(h * 0.72)
        roi_x1, roi_x2 = int(w * 0.1), int(w * 0.9)
        mask_roi = np.zeros_like(mask)
        mask_roi[roi_y1:roi_y2, roi_x1:roi_x2] = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        mask_roi = cv2.medianBlur(mask_roi, 3)

        # Contour detection
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mug_center = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if 10 < area < 350:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mug_center = (cx, cy)

        # Hand zones
        left_zone = (int(w * 0.15), int(h * 0.55), int(w * 0.54), int(h * 0.7))
        right_zone = (int(w * 0.55), int(h * 0.55), int(w * 0.85), int(h * 0.7))

        # Determine which hand zone is relevant
        if mug_center is not None:
            if mug_center[0] < w // 2:
                hand_zone = right_zone
            else:
                hand_zone = left_zone
        else:
            hand_zone = right_zone

        hx1, hy1, hx2, hy2 = hand_zone

        # Proximity test
        in_hand_zone = False
        proximity_val = 0.0
        if mug_center is not None:
            x, y = mug_center
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                in_hand_zone = True
                proximity_val = 1.0
            else:
                hand_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
                dist = np.linalg.norm(np.array(mug_center) - np.array(hand_center))
                proximity_val = max(0.0, 1.0 - (dist / 70.0))

        return mug_center, mask_roi, in_hand_zone, proximity_val

    # Gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_count = 0
        self.has_tried_catch = False
        self.win_detected = False
        self.lose_detected = False
        self.game_active = False
        self.color_window.clear()
        self.absent_streak = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        obs_f = obs.astype(np.float32)
        total_reward = -0.005

        # Detect A press
        pressed_a = False
        if isinstance(action, np.ndarray):
            pressed_a = len(action) > self.BUTTON_A and action[self.BUTTON_A] == 1
        elif isinstance(action, int):
            pressed_a = action == self.BUTTON_A

        # Mug detection
        mug_center, mask, in_hand_zone, proximity_val = self._find_mug(obs)
        mug_visible = mug_center is not None

        # Handle visual freeze timer
        if hasattr(self, "freeze_timer") and self.freeze_timer > 0:
            self.freeze_timer -= 1
            info["reward"] = 0.0
            return obs, 0.0, terminated, truncated, info

        # Background color detection (win/lose)
        r, g, b = self._region_rgb_mean(obs_f, self.play_region)
        current_color = np.array([r, g, b])

        # Reference target colors
        win_color = np.array([248, 248, 8])
        lose_color = np.array([0, 0, 0])

        # Euclidean color distance
        dist_win = np.linalg.norm(current_color - win_color)
        dist_lose = np.linalg.norm(current_color - lose_color)

        # Reset win/lose flags when background returns to normal
        if (self.win_detected or self.lose_detected) and dist_win >= 40 and dist_lose >= 40:
            self.win_detected = False
            self.lose_detected = False
            self.has_tried_catch = False
            self.game_active = False
            self.absent_streak = 0
            self.freeze_timer = 0

        # Manage gameplay state
        if mug_visible:
            self.absent_streak = 0

            # Only start gameplay when not in win/lose transition
            if not self.game_active and not (self.win_detected or self.lose_detected):
                self.game_active = True
                self.has_tried_catch = False
        else:
            self.absent_streak += 1
            if self.absent_streak > 10:
                self.game_active = False
                self.has_tried_catch = False
                return obs, 0.0, terminated, truncated, info

        # Round-end detection
        if self.game_active:
            # WIN detection
            if dist_win < 40 and not self.win_detected:
                if self.debug:
                    print(f"[ROUND END] WIN at frame {self.frame_count}, RGB={current_color}", flush=True)
                total_reward = +3.0
                self.win_detected = True
                self.lose_detected = False
                self.game_active = False
                self.freeze_timer = 60

            # LOSE detection
            elif dist_lose < 40 and not self.lose_detected:
                if self.debug:
                    print(f"[ROUND END] LOSE at frame {self.frame_count}, RGB={current_color}", flush=True)
                total_reward = -5.0
                self.lose_detected = True
                self.win_detected = False
                self.game_active = False
                self.freeze_timer = 60

            # Active round reward shaping
            else:
                if mug_visible:
                    total_reward += 0.02 + proximity_val * 0.1

                    # A-press reward scaling + early penalty
                    if pressed_a and not self.has_tried_catch:
                        if proximity_val < 0.3:
                            total_reward -= 0.11

                        a_press_bonus = 0.3 * min(1.0, proximity_val / 0.8)
                        total_reward += a_press_bonus
                        self.has_tried_catch = True

                        if self.debug:
                            print(f"[CATCH ATTEMPT] Frame {self.frame_count} (A-press reward={a_press_bonus:.2f})")

                    # Extra reward for grabbing within hand zone
                    if in_hand_zone and self.has_tried_catch:
                        total_reward += 0.5
                        info["catch_signal"] = True

                else:
                    if not self.has_tried_catch:
                        total_reward -= 0.005

        # Update info dict
        info.update({
            "reward": total_reward,
            "mug_center": mug_center,
            "in_hand_zone": in_hand_zone,
            "proximity": proximity_val,
            "win": self.win_detected,
            "lose": self.lose_detected,
            "game_active": self.game_active,
            "has_tried_catch": self.has_tried_catch,
            "rgb": current_color,
        })

        return obs, total_reward, terminated, truncated, info
