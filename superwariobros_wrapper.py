import gymnasium as gym
import numpy as np
import cv2


class SuperWarioBrosRewardWrapper(gym.Wrapper):
    """
    Visual reward shaping for 'Super Wario Bros' microgame. Detects entry/exit using background
    color patches, counts Goombas, and applies reward shaping by marking # of goombas squished
    and whether Wario squished all of them within the time limit.
    """
    def __init__(self, env):
        super().__init__(env)
        self.allowed_actions = [0, 2, 3]

        self.in_microgame = False
        self.prev_in_microgame = False
        self.prev_goombas = 0
        self.loss_given = False
        self.win_given = False

        # Patch coordinates (for detecting microgame area)
        self.patch1_y1, self.patch1_y2 = 10, 19
        self.patch1_x1, self.patch1_x2 = 19, 26
        self.patch2_y1, self.patch2_y2 = 10, 19
        self.patch2_x1, self.patch2_x2 = 11, 18


    # Detect whether we're inside the microgame
    def in_microgame_check(self, frame_rgb):
        small = cv2.resize(frame_rgb, (100, 100))
        patch1 = small[self.patch1_y1:self.patch1_y2, self.patch1_x1:self.patch1_x2]
        patch2 = small[self.patch2_y1:self.patch2_y2, self.patch2_x1:self.patch2_x2]

        if patch1.size == 0 or patch2.size == 0:
            return False

        # Sky blue (R,G,B) ≈ (152, 192, 248)
        lower_blue = np.array([110, 150, 210], dtype=np.uint8)
        upper_blue = np.array([190, 230, 255], dtype=np.uint8)

        mask1 = cv2.inRange(patch1, lower_blue, upper_blue)
        mask2 = cv2.inRange(patch2, lower_blue, upper_blue)

        return np.mean(mask1 > 0) > 0.5 and np.mean(mask2 > 0) > 0.5


    # Count Goombas
    def count_goombas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Broad orange/brown detection tuned for 100x100 downscale
        lower_brown = np.array([4, 40, 30], dtype=np.uint8)
        upper_brown = np.array([32, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Exclude non-goomba regions
        h, w = mask.shape
        sky_cut = int(h * 0.72)
        floor_cut = int(h * 0.80)
        mask[:sky_cut, :] = 0
        mask[floor_cut:, :] = 0

        # Detect Wario’s colors (purple and yellow) and remove them
        lower_purple = np.array([130, 80, 60], dtype=np.uint8)
        upper_purple = np.array([165, 255, 255], dtype=np.uint8)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        lower_yellow = np.array([20, 80, 100], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine and subtract from Goomba mask
        wario_mask = cv2.bitwise_or(mask_purple, mask_yellow)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(wario_mask))


        # Morphological cleanup
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Blob filtering to find goombas
        goomba_count = 0
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            area = w_box * h_box
            aspect = w_box / float(h_box + 1e-6)
            if 25 <= area <= 60 and 0.6 <= aspect <= 1.1 and 70 <= y <= 78:
                goomba_count += 1

        return goomba_count, mask


    #Gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.in_microgame = False
        self.prev_goombas = 0
        self.loss_given = False
        self.win_given = False
        return obs, info


    def step(self, action):
        # Restrict illegal inputs so DQN doesn't press A, UP, or DOWN
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        if action not in self.allowed_actions:
            action = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = cv2.resize(obs, (100, 100))
        shaped_reward = 0.0

        in_microgame = self.in_microgame_check(frame)

        # Detect transitions
        entering = (not self.prev_in_microgame) and in_microgame
        exiting = self.prev_in_microgame and not in_microgame

        if entering:
            # Reset counters for new round
            self.prev_goombas = 0
            self.loss_given = False
            self.win_given = False

        if in_microgame:
            goombas, _ = self.count_goombas(frame)

            if self.prev_goombas == 0:
                self.prev_goombas = goombas

            # Goomba squished = small reward
            if goombas < self.prev_goombas:
                shaped_reward += (self.prev_goombas - goombas) * 1.0

            # All goombas cleared = one-time big reward
            if goombas == 0 and not self.win_given:
                shaped_reward += 5.0
                self.win_given = True

            self.prev_goombas = goombas

        elif exiting:
            # Apply one-time failure penalty if goombas remain
            if self.prev_goombas > 0 and not self.loss_given:
                shaped_reward -= 5.0
                self.loss_given = True

        # Save state for next frame
        self.prev_in_microgame = in_microgame
        self.in_microgame = in_microgame

        total_reward = float(reward + shaped_reward)
        info.update({
            "in_microgame": in_microgame,
            "goombas": self.prev_goombas,
            "shaped_reward": shaped_reward
        })
        return obs, total_reward, terminated, truncated, info
