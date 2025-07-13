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
        time.sleep(0.2) # Small delay after pressing key to allow game to register
        self.keyboard.release(self.key_map[action]) # Release the key after the delay
        time.sleep(self.step_duration - 0.2) # Remaining step duration


        state = self.get_state()
        reward, done = self.compute_reward(state, action)

        self.total_steps += 1
        if self.total_steps >= self.max_steps:
            done = True

        return state, reward, done, {}

    def press_key(self, action):
        self.key_map = {'up': Key.up, 'left': Key.left, 'right': Key.right, 'enter': Key.enter}
        key = self.key_map[action]
        self.keyboard.press(key)
        
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

# PSO
def evaluate_sequence(env, sequence):
    state = env.reset()
    total_reward = 0
    done = False

    for action_index in sequence:
        if done:
            break
        state, reward, done, _ = env.step(int(round(action_index))) # PSO might give float values, convert to int
        total_reward += reward
    return total_reward

class Particle:
    def __init__(self, sequence_length, action_space_size):
        self.position = np.random.randint(0, action_space_size, size=sequence_length).astype(float)
        self.velocity = np.random.uniform(-1, 1, size=sequence_length)
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

def pso_optimize(env, generations=20, num_particles=15, sequence_length=40,
                 c1=2.0, c2=2.0, w=0.7):
    action_space_size = env.get_action_space().n
    particles = [Particle(sequence_length, action_space_size) for _ in range(num_particles)]
    global_best_position = None
    global_best_score = -float('inf')

    for gen in range(generations):
        for particle in particles:
            current_score = evaluate_sequence(env, particle.position)

            if current_score > particle.best_score:
                particle.best_score = current_score
                particle.best_position = np.copy(particle.position)

            if current_score > global_best_score:
                global_best_score = current_score
                global_best_position = np.copy(particle.position)

        for particle in particles:
            r1 = np.random.uniform(0, 1, size=sequence_length)
            r2 = np.random.uniform(0, 1, size=sequence_length)

            cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
            social_velocity = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
            
            # Clamp velocities to a reasonable range
            particle.velocity = np.clip(particle.velocity, -10, 10) # Adjust max velocity as needed

            particle.position = particle.position + particle.velocity
            
            # Ensure actions are within the valid range [0, action_space_size - 1]
            particle.position = np.round(np.clip(particle.position, 0, action_space_size - 1))

        print(f"Generation {gen+1}: Global Best Score = {global_best_score} | Global Best Sequence = {[int(x) for x in global_best_position]}")
    return global_best_position, global_best_score

# run
if __name__ == '__main__':
    env = TrackmaniaRTEnv()
    best_sequence, best_score = pso_optimize(env)
    print(f"\nOptimization Finished! Best sequence found: {[int(x) for x in best_sequence]} with score: {best_score}")