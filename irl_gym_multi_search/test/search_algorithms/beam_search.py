import numpy as np
import heapq

def simulate_step(size_x, size_y, position, action, penalty):
    # Define the effects of each action
    action_effects = [
        np.array([0, -1]),  # Move up
        np.array([-1, 0]),  # Move left
        np.array([0, 1]),   # Move down
        np.array([1, 0]),   # Move right
    ]
    new_position = position + action_effects[action]
    
    # Check if the new position is outside the grid
    outside = np.any(new_position < [0, 0]) or np.any(new_position >= [size_x, size_y])
    
    # Clip the new position to ensure it stays within bounds
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    
    # Apply penalty if the position is outside the grid
    position_penalty = penalty if outside else 0
    
    return tuple(new_position), position_penalty

def beam_search(size_x, size_y, cur_pos, distribution, steps=5, beam_width=10, penalty=-5):
    # Initialize the beam with the starting position and a score of 0
    beam = [(0, [cur_pos])]

    for _ in range(steps):
        new_beam = []
        for score, path in beam:
            position = path[-1]
            for action in range(4):
                new_position, position_penalty = simulate_step(size_x, size_y, position, action, penalty)
                new_score = score + distribution[new_position[0], new_position[1]] + position_penalty
                new_path = path + [new_position]
                new_beam.append((new_score, new_path))
        
        # Keep only the top beam_width paths based on their scores
        beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])

    best_score, best_path = beam[0]
    return best_path[1]  # return the next position in the best path
