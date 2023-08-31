import numpy as np

class Node:
    def __init__(self, state):
        self.state = state
        self.children = []
        self.parent = None
        self.value = 0
        self.visits = 0
        self.action_taken_to_reach = None

    def update_value(self, simulation_result):
        self.value += simulation_result
        self.visits += 1


class RegionMCTSSearch:
    def __init__(self, mcts_iterations=250, c_param=0.95, simulation_steps=30):
        self.visited_cells = set()
        self.current_region = None
        self.searched_regions = set()
        self.last_region_index = None
        self.all_regions_searched = False
        self.mcts_iterations = mcts_iterations
        self.c_param = c_param
        self.simulation_steps = simulation_steps
        self.last_action = None
        self.orientation_mapping = {0: 'u', 1: 'r', 2: 'd', 3: 'l'}

    def perform_mcts_within_region(self, x, y, search_distribution, fov_dict):
        direction = {0: 'u', 1: 'r', 2: 'd', 3: 'l', 4: 'u'}[self.last_action % 5] if self.last_action is not None else 'u'
        state = (x, y, direction)
        root = Node(state)
        for _ in range(self.mcts_iterations):
            node = self.tree_policy(root, search_distribution)
            simulation_result = self.simulate_steps(node, search_distribution)
            self.backpropagation(node, simulation_result)

        best_action = self.select_best_action(root)
        self.last_action = best_action
        return best_action

    def tree_policy(self, node, search_distribution):
        # Traverse the tree until we reach a leaf node (one without children)
        while node.children:
            if len(node.children) < 4: # If not all actions have been explored, expand the current node
                return self.expand(node, search_distribution)
            else: # Otherwise, proceed to the best child node
                node = self.select_best_child(node, exploration_weight=self.c_param)
        return self.expand(node, search_distribution)

    def expand(self, node, search_distribution):
        available_actions = set(range(4)) - {child.action_taken_to_reach for child in node.children}
        action = available_actions.pop() # Select any unexplored action
        x, y, direction = node.state
        new_x, new_y, new_direction = self.apply_action(x, y, direction, action)
        new_state = (new_x, new_y, new_direction)
        child = Node(new_state)
        child.action_taken_to_reach = action
        child.parent = node
        node.children.append(child)
        return child

    def backpropagation(self, node, simulation_result):
        while node:
            node.update_value(simulation_result)
            node = node.parent

    def select_best_child(self, node, exploration_weight):
        best_value = -float('inf')
        best_child = None
        for child in node.children:
            ucb_value = (child.value / child.visits) + exploration_weight * np.sqrt(np.log(node.visits) / child.visits)
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        return best_child

    def simulate_steps(self, node, search_distribution):
        x, y, direction = node.state
        total_value = 0
        max_x, max_y = search_distribution.shape
        unvisited_bias = 1.0  # Bias towards unvisited cells
        outside_region_penalty = -1000.0  # Significant penalty for going outside the region
        high_value_factor = 1000.0  # Factor to emphasize higher-valued cells
            
        for _ in range(self.simulation_steps):
            # print("Current Position:", (x, y)) # Print the current position
            neighbor_values = []
            for action in range(4): # Iterate through actions to evaluate neighbor cells
                dx, dy = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}[action]
                nx, ny = x + dx, y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:  # Check if the neighbor is within bounds
                    value = search_distribution[nx, ny] * high_value_factor  # Multiply by the factor
                    if (nx, ny) not in self.visited_cells:
                        value += unvisited_bias  # Add bias for unvisited cells
                    if not self.in_region(nx, ny):  # Check if outside the current region
                        value += outside_region_penalty  # Apply significant penalty if outside the region
                    neighbor_values.append(value)
                else:
                    neighbor_values.append(outside_region_penalty)  # Assign the penalty for out-of-bounds neighbors

            action = np.argmax(neighbor_values)
            new_x, new_y, new_direction = self.apply_action(x, y, direction, action)

            x, y, direction = new_x, new_y, new_direction
            step_value = search_distribution[x, y]
            if not self.in_region(x, y):  # If the new position is outside the current region, apply the penalty
                step_value = outside_region_penalty
            total_value += step_value

            # print("Neighbor Values:", neighbor_values) # Print the neighbor values
            # print("Chosen Action:", action) # Print the chosen action
            # print("New Position:", (new_x, new_y), "In Region:", self.in_region(new_x, new_y)) # Print the new position and whether it's inside the region
        return total_value

    def apply_action(self, x, y, direction, action):
        # Since the action dictates the movement, use the action list mapping
        dx, dy = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}[action]
        new_x, new_y = x + dx, y + dy

        # Check if the new position is within the current region
        if not self.in_region(new_x, new_y):
            # If the action leads outside the region, retain the current position and direction
            return x, y, direction

        # Update the orientation based on the action taken
        new_direction = {0: 'u', 1: 'r', 2: 'd', 3: 'l'}[action]

        return new_x, new_y, new_direction

    def mark_visited_cells(self, x, y, fov):
        # Mark cells as visited based on the custom FOV
        fov_agent_x, fov_agent_y = fov["fov_agent_position"]
        for dx, row in enumerate(fov["fov"]):
            for dy, cell in enumerate(row):
                if cell == 1:
                    visit_x = x + dx - fov_agent_x
                    visit_y = y + dy - fov_agent_y
                    self.visited_cells.add((visit_x, visit_y))

    def select_next_action(self, cur_pos, distribution, fov_dict, current_fov, regions):
        x, y = cur_pos
        return self.select_next_action_internal(x, y, distribution, regions, fov_dict, current_fov)

    def select_next_action_internal(self, x, y, search_distribution, regions, fov_dict, current_fov):
        fov = current_fov
        fov_size_x, fov_size_y = fov_dict["fov_size"]
        fov_agent_x, fov_agent_y = fov_dict["fov_agent_position"]
        for i in range(fov_size_y): # Mark all cells in the FOV as visited
            for j in range(fov_size_x):
                if fov[i][j] == 1:
                    cell_x = x + (j - fov_agent_x)
                    cell_y = y + (i - fov_agent_y)
                    self.visited_cells.add((cell_x, cell_y))

        # Check if current region is fully visited, and if so, reset it
        if self.current_region and not any(tuple(p) not in self.visited_cells for p in self.current_region['points']):
            if self.current_region['index'] != self.last_region_index:
                print(f"Region {self.current_region['index']} has no unvisited points left.")
                self.last_region_index = self.current_region['index']
            self.searched_regions.add(self.current_region['index'])
            self.current_region = None

        # If the agent is not in a region, select the next best region
        if self.current_region is None:
            self.current_region = self.select_next_region(regions, x, y)
            if self.current_region:
                if self.current_region['index'] != self.last_region_index:
                    print(f"Going to region {self.current_region['index']}.") # Navigating to a new region
                    self.last_region_index = self.current_region['index']
                return self.get_action_to_navigate_to_region(x, y)
            else:
                if not self.all_regions_searched:
                    print('All regions searched. Searching the entire environment.')
                    self.all_regions_searched = True
                return self.search_entire_environment(x, y, search_distribution)

        # Check if the current position is within the current region
        if (x, y) in map(tuple, self.current_region['points']):
            # Print the update for cells searched within the region
            cells_searched = sum(1 for p in self.current_region['points'] if tuple(p) in self.visited_cells)
            total_cells = len(self.current_region['points'])
            print(f"Region {self.current_region['index']} searched {cells_searched}/{total_cells} cells.") # Update on cells searched
            return self.get_action_within_region(x, y, search_distribution, fov_dict)
        else:
            unvisited_points = [p for p in self.current_region['points'] if (tuple(p) not in self.visited_cells)]
            unvisited_points.sort(key=lambda p: -search_distribution[p[0], p[1]])
            return self.get_action_to_navigate_to_cell(x, y, unvisited_points[0])

    def select_next_region(self, regions, current_x, current_y):
        def region_priority(region):
            unvisited_points = [p for p in region['points'] if (tuple(p) not in self.visited_cells)]
            # Add a threshold to ensure only regions with sufficient unvisited points are considered
            if len(unvisited_points) < 5:
                return 0
            centroid_x, centroid_y = region['centroid']
            distance = np.sqrt((current_x - centroid_x)**2 + (current_y - centroid_y)**2)
            weight_factor = region['weight'] * 0.01
            distance_factor = 1 / (1 + distance) * 1
            return weight_factor * distance_factor

        unsearched_regions = [r for r in regions if r['index'] not in self.searched_regions]
        unsearched_regions.sort(key=region_priority, reverse=True)
        return unsearched_regions[0] if unsearched_regions else None

    def get_action_within_region(self, x, y, search_distribution, fov_dict):
        # ... handle boundary cases if needed ...

        # Perform MCTS within the region
        best_action = self.perform_mcts_within_region(x, y, search_distribution, fov_dict)

        # Optionally, add some logic to navigate towards higher-value cells if MCTS action is not promising
        # ...

        return best_action

    def get_action_to_navigate_to_region(self, x, y):
        target_x, target_y = self.current_region['peak_x'], self.current_region['peak_y']
        return self.get_action_to_navigate_to_cell(x, y, [target_x, target_y])

    def get_action_to_navigate_to_cell(self, x, y, target_point):
        target_x, target_y = target_point
        relative_pos = [target_x - x, target_y - y]

        if np.abs(relative_pos[0]) >= np.abs(relative_pos[1]):
            if relative_pos[0] < 0:
                action = 1  # Move left
            else:
                action = 3  # Move right
        else:
            if relative_pos[1] < 0:
                action = 0  # Move up
            else:
                action = 2  # Move down

        if relative_pos[0] == 0 and relative_pos[1] == 0:
            action = 4

        return action

    def search_entire_environment(self, x, y, search_distribution):
        max_value = -float('inf')
        target_point = None
        for i in range(search_distribution.shape[0]):
            for j in range(search_distribution.shape[1]):
                if (i, j) not in self.visited_cells and search_distribution[i, j] > max_value:
                    max_value = search_distribution[i, j]
                    target_point = [i, j]
        return self.get_action_to_navigate_to_cell(x, y, target_point)

    def select_best_action(self, root):
        best_child = self.select_best_child(root, exploration_weight=0)  # Exploitation only
        best_action = best_child.action_taken_to_reach # Assuming this attribute holds the action taken to reach the child
        return best_action

    def in_region(self, x, y):
        return (x, y) in map(tuple, self.current_region['points'])

