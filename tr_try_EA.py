import numpy as np
import random
import time
import cv2
import mss
from pynput.keyboard import Key, Controller
from rtgym import RealTimeGymInterface 
import gym


class TrackmaniaRTEnv(RealTimeGymInterface):
    def __init__(self):
        self.actions = ['left', 'right', 'up']
        self.keyboard = Controller()
        self.step_duration = 0.3
        self.max_steps = 50
        self.total_steps = 0
        self.capture_size = (42, 42)
        self.capture_region = {"top": 300, "left": 700, "width": 500, "height": 300}
        self.sct = mss.mss()
        self.last_frame = None
        self.same_frame_count = 0
        self.stationary_threshold = 5
        self.enter_pressed = False

    def get_observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(42*42,), dtype=np.uint8)

    def get_action_space(self):
        return gym.spaces.Discrete(len(self.actions))  # 0, 1, 2

    def reset(self):
        self.total_steps = 0
        self.last_frame = None
        self.same_frame_count = 0
        self.enter_pressed = False
        time.sleep(2)
        return self.get_state()

    def step(self, action_index):
        action = self.actions[action_index]
        is_stuck = self.same_frame_count >= self.stationary_threshold

        if is_stuck and not self.enter_pressed:
            self.press_key('enter')
            self.enter_pressed = True
            time.sleep(self.step_duration)
            reward = -5
            done = False
            state = self.get_state()
            return state, reward, done, {}

        self.press_key(action)
        time.sleep(self.step_duration)

        state = self.get_state()
        reward, done = self.compute_reward(state, action)

        self.total_steps += 1
        if self.total_steps >= self.max_steps:
            done = True

        return state, reward, done, {}

    def press_key(self, action):
        key_map = {'up': Key.up, 'left': Key.left, 'right': Key.right, 'enter': Key.enter}
        key = key_map[action]
        self.keyboard.press(key)
        time.sleep(0.2)
        self.keyboard.release(key)

    def get_state(self):
        screen = self.sct.grab(self.capture_region)
        img = np.array(screen)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img, self.capture_size)
        return img.flatten()

    def compute_reward(self, current_frame, action):
        if self.last_frame is not None:
            diff = np.abs(current_frame.astype(np.int16) - self.last_frame.astype(np.int16))
            change = np.mean(diff)
            if change < 2.0:
                self.same_frame_count += 1
            else:
                self.same_frame_count = 0
        else:
            self.same_frame_count = 0

        self.last_frame = current_frame

        if self.same_frame_count < self.stationary_threshold:
            self.enter_pressed = False

        if self.same_frame_count >= self.stationary_threshold:
            return -10, False
        elif self.same_frame_count == 0:
            return 5, False
        else:
            return 1, False

#EA
def evaluate_sequence(env, sequence):
    state = env.reset()
    total_reward = 0
    done = False

    for action_index in sequence:
        if done:
            break
        state, reward, done, _ = env.step(action_index)
        total_reward += reward

    return total_reward


def mutate(sequence, action_space_size, mutation_rate=0.2):
    return [random.randint(0, action_space_size - 1) if random.random() < mutation_rate else act for act in sequence]


def evolve(env, generations=20, population_size=15, sequence_length=40):
    population = [random.choices(range(env.get_action_space().n), k=sequence_length) for _ in range(population_size)]

    for gen in range(generations):
        scores = [evaluate_sequence(env, seq) for seq in population]
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score, best_seq = ranked[0]

        print(f"Generation {gen+1}: Best Score = {best_score} | Best Sequence = {best_seq}")

        next_gen = [best_seq]
        while len(next_gen) < population_size:
            parent = random.choice(ranked[:5])[1]
            child = mutate(parent, env.get_action_space().n)
            next_gen.append(child)

        population = next_gen

#run
if __name__ == '__main__':
    env = TrackmaniaRTEnv()
    evolve(env)