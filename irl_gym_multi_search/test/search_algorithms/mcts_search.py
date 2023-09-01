
import numpy as np
from queue import PriorityQueue
from concurrent.futures import ProcessPoolExecutor
import os

class MCTSNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.action = None
        self.ucb = np.inf  # Initially set to infinity to ensure this node is selected first

    def expand(self, priors):
        for i, prior in enumerate(priors):
            child = MCTSNode(self, prior)
            child.action = i
            self.children.append(child)

class MCTS:
    def __init__(self, size_x, size_y, cur_pos, distribution, fov_dict, current_fov, cur_orientation, last_action, obstacles, num_iterations, c_param, max_rollout_steps=150):
        self.size_x = size_x
        self.size_y = size_y
        self.cur_pos = cur_pos
        self.last_action = last_action
        self.current_fov = current_fov
        self.fov_offsets = [(i - len(current_fov) // 2, j - len(current_fov) // 2) for i in range(len(current_fov)) for j in range(len(current_fov)) if current_fov[i, j] == 1]
        self.orientation = cur_orientation
        self.orientation_mapping = {'u': 0, 'r': 1, 'd': 2, 'l': 3}
        self.distribution = distribution
        self.distribution /= np.sum(self.distribution)
        self.obstacles = obstacles
        self.num_iterations = num_iterations
        self.c_param = c_param
        self.max_rollout_steps = max_rollout_steps
        self.actions = {
            0: np.array([0, -1]),  # up
            1: np.array([-1, 0]),  # left
            2: np.array([0, 1]),  # down
            3: np.array([1, 0]),  # right
        }
        self.total_visits = 0
        root_priors = self.compute_priors(self.cur_pos, self.current_fov)
        self.root = MCTSNode(None, 1.0)  # Create the root node
        self.root.expand(root_priors)  # Expand the root's children

    def action_to_pos(self, action, cur_pos):
        # Convert the resulting array to a tuple of integers
        next_pos = tuple(cur_pos + self.actions[action])
        return next_pos, [action]

    def compute_ucb(self, node):
        if node.visits == 0 or node.visits != node.parent.visits:
            # Using PUCT formula for UCB
            node.ucb = node.value / (1 + node.visits) + self.c_param * node.prior * np.sqrt(node.parent.visits + 1) / (1 + node.visits)
        return node.ucb

    def compute_reward(self, next_pos, fov):
        reward = 0
        if 0 <= next_pos[0] < self.size_x and 0 <= next_pos[1] < self.size_y:
            for offset in self.fov_offsets:
                relative_pos = next_pos + offset
                if (0 <= relative_pos[0] < self.size_x and 0 <= relative_pos[1] < self.size_y and self.obstacles[relative_pos[0], relative_pos[1]] != 1):
                    reward += self.distribution[relative_pos[0], relative_pos[1]] ** 2
        else:
            reward = -np.inf
        return reward

    def compute_priors(self, cur_pos, fov):
        action_rewards = []
        
        for action in self.actions:
            delta = self.actions[action]
            next_pos = cur_pos + delta
            reward = self.compute_reward(next_pos, fov)
            action_rewards.append(reward)

        max_reward = max(action_rewards)
        action_rewards = [r if r != -np.inf else 0 for r in action_rewards]
        total = sum(action_rewards)
        action_probabilities = [r / total if total != 0 else 1.0 / len(self.actions) for r in action_rewards]
        return action_probabilities

    def select_child(self, node):
        best_score = -np.inf
        best_child = None
        for child in node.children:
            score = self.compute_ucb(child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self, action, cur_pos, fov):
        reward = 0
        for step in range(self.max_rollout_steps):
            next_pos, path_or_action = self.action_to_pos(action, cur_pos)
            if next_pos[0] < 0 or next_pos[0] >= self.size_x or next_pos[1] < 0 or next_pos[1] >= self.size_y:
                continue
            if self.in_fov(next_pos, fov, cur_pos):
                reward += self.distribution[next_pos[0]][next_pos[1]]
            cur_pos = next_pos
        return reward

    def get_best_action(self, cur_pos):
        action_probabilities = self.compute_priors(cur_pos)
        return np.random.choice(len(self.actions), p=action_probabilities)

    def backpropagate(self, path, reward):
        for node in reversed(path):
            node.visits += 1
            node.value += reward
            reward *= 0.99  # Discount factor for multi-step ahead reward

    def in_fov(self, pos, fov, cur_pos):
        relative_pos = np.array(pos) - np.array(cur_pos)
        fov_center = np.array(fov.shape) // 2
        fov_pos = fov_center + relative_pos

        if 0 <= fov_pos[0] < fov.shape[0] and 0 <= fov_pos[1] < fov.shape[1]:
            in_fov = fov[fov_pos[0], fov_pos[1]] == 1
            return in_fov
        return False

    def tree_search(self, node):
        path = [node]
        cur_pos = np.array(self.cur_pos).flatten()[:2]
        fov = self.current_fov
        rotation_degree = 0
        while len(path[-1].children) != 0:
            node = self.select_child(path[-1])
            next_pos, path_to_region = self.action_to_pos(node.action, cur_pos) # Unpack the return value here
            cur_pos = next_pos # Update cur_pos with the new position
            action_to_rotation = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}
            if node.action < 5:
                rotation_degree = action_to_rotation[node.action]
            fov = np.rot90(fov, k=rotation_degree)

            if not (0 <= cur_pos[0] < self.size_x and 0 <= cur_pos[1] < self.size_y):
                break
            path.append(node)
        if path[-1] is self.root:
            priors = self.compute_priors(cur_pos, fov)
            path[-1].expand(priors)
            node = self.select_child(path[-1])
            reward = self.rollout(node.action, cur_pos, fov)
            return node, reward
        elif path[-1].visits == 0:
            reward = self.rollout(path[-1].action, cur_pos, fov)
            return path[-1], reward
        else:
            priors = self.compute_priors(cur_pos, fov)
            path[-1].expand(priors)
            node = self.select_child(path[-1])
            reward = self.rollout(node.action, cur_pos, fov)
            return node, reward

    def simulate(self):
        rewards = []

        for _ in range(self.num_iterations):
            node, reward = self.tree_search(self.root)
            rewards.append(reward)
            path = [node]
            while node is not self.root:
                node = node.parent
                path.append(node)
            self.backpropagate(path, reward)
            self.total_visits += 1

        best_action_node = self.select_child(self.root)
        best_action = best_action_node.action
        max_reward = max(rewards)
        if best_action > 4:
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        best_path = [best_action]
        return best_path