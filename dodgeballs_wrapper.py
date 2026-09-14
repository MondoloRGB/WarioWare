import gymnasium as gym
import numpy as np
import cv2

class DodgeBallsRewardWrapper(gym.Wrapper):
    """
        Reward shaping for the 'Dodge Balls' microgame. Detects entry/exit using background
        color patches, acquires a mask of the car, and applies reward shaping based on how long it
        can survive within the frame (by not disappearing from getting crushed).
    """
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.allowed_actions = [0, 2, 3, 4, 5]
        self.debug = debug

        self.in_microgame = False
        self.death_registered = False

        # Patch coordinates (for detecting microgame area)
        self.patch1_y1, self.patch1_y2 = 6, 15
        self.patch1_x1, self.patch1_x2 = 78, 87

        self.patch2_y1, self.patch2_y2 = 6, 15
        self.patch2_x1, self.patch2_x2 = 10, 19

    # Detect whether we're inside the microgame
    def in_microgame_check(self, frame_rgb):
        small = cv2.resize(frame_rgb, (100, 100))
        patch1 = small[self.patch1_y1:self.patch1_y2, self.patch1_x1:self.patch1_x2]
        patch2 = small[self.patch2_y1:self.patch2_y2, self.patch2_x1:self.patch2_x2]

        # Green range for the Region of Interest (ROI)
        lower_green = np.array([60, 160, 0])
        upper_green = np.array([100, 210, 40])

        mask1 = cv2.inRange(patch1, lower_green, upper_green)
        mask2 = cv2.inRange(patch2, lower_green, upper_green)

        green_ratio1 = np.mean(mask1 > 0)
        green_ratio2 = np.mean(mask2 > 0)

        in_game = green_ratio1 > 0.2 and green_ratio2 > 0.2

        return in_game

    # Detect car mask
    def get_car_mask(self, frame_rgb):
        small = cv2.resize(frame_rgb, (100, 100))

        # Red tones (car body)
        lower_red1 = np.array([190, 0, 0])
        upper_red1 = np.array([255, 80, 80])
        red_mask = cv2.inRange(small, lower_red1, upper_red1)

        # Yellow tones (car trim)
        lower_yellow = np.array([200, 180, 80])
        upper_yellow = np.array([255, 230, 130])
        yellow_mask = cv2.inRange(small, lower_yellow, upper_yellow)

        # Combine both
        mask = cv2.bitwise_or(red_mask, yellow_mask)
        return mask

    # Gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.in_microgame = False
        self.death_registered = False
        self.prev_ratio = 0.0
        self.prev_visible = False
        return obs, info

    def step(self, action):
        # Restrict illegal inputs so DQN doesn't press A
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        if action not in self.allowed_actions:
            action = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        frame_rgb = obs
        shaped_reward = 0.0

        # Check microgame state
        in_microgame = self.in_microgame_check(frame_rgb)
        self.in_microgame = in_microgame

        # Car detection
        car_visible = False
        visible_ratio = 0.0

        if in_microgame:
            mask = self.get_car_mask(obs)
            visible_ratio = float(np.mean(mask > 0))

            # Smooth transitions
            self.prev_ratio = getattr(self, "prev_ratio", 0.0)
            smoothed_ratio = 0.7 * self.prev_ratio + 0.3 * visible_ratio
            self.prev_ratio = smoothed_ratio

            # Hysteresis
            self.prev_visible = getattr(self, "prev_visible", False)
            appear_thresh = 0.0008
            disappear_thresh = 0.0005

            if smoothed_ratio > appear_thresh:
                car_visible = True
            elif smoothed_ratio < disappear_thresh:
                car_visible = False
            else:
                car_visible = self.prev_visible

            self.prev_visible = car_visible

        info["car_visible"] = car_visible
        if self.debug:
            pratio = getattr(self, "prev_ratio", 0.0)
            print(f"[DEBUG] In microgame: {in_microgame} | Car visible: {car_visible} | "
                  f"Visible ratio: {visible_ratio:.5f} | Smoothed: {pratio:.5f}")

        # Reward shaping & death logic
        if in_microgame:
            self.invisible_frames = getattr(self, "invisible_frames", 0)
            if not car_visible:
                self.invisible_frames += 1
            else:
                self.invisible_frames = 0

            max_invisible = 4
            if self.invisible_frames > max_invisible and not self.death_registered:
                shaped_reward -= 1.0
                self.death_registered = True
                info["car_dead"] = True

            elif car_visible and not self.death_registered:
                shaped_reward += 0.1
                info["car_dead"] = False

        else:
            self.death_registered = False
            self.invisible_frames = 0

        # Combine rewards
        total_reward = float(reward + shaped_reward)
        info["shaped_reward"] = shaped_reward

        return obs, total_reward, terminated, truncated, info
