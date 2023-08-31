import numpy as np

class RegionGreedySearch:
    def __init__(self):
        self.visited_cells = set()
        self.current_region = None
        self.searched_regions = set()
        self.last_region_index = None
        self.all_regions_searched = False


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
                    print(f"Going to region {self.current_region['index']}.")
                    self.last_region_index = self.current_region['index']
                return self.get_action_to_navigate_to_region(x, y)
            else:
                if not self.all_regions_searched:
                    print('All regions searched. Searching the entire environment.')
                    self.all_regions_searched = True
                return self.search_entire_environment(x, y, search_distribution)

        # Check if the current position is within the current region
        if (x, y) in map(tuple, self.current_region['points']):
            return self.get_action_within_region(x, y, search_distribution, fov_dict) # Added fov_dict here
        else:
            unvisited_points = [p for p in self.current_region['points'] if (tuple(p) not in self.visited_cells)]
            unvisited_points.sort(key=lambda p: -search_distribution[p[0], p[1]])
            return self.get_action_to_navigate_to_cell(x, y, unvisited_points[0])


    def select_next_region(self, regions, current_x, current_y):
        def region_priority(region):
            centroid_x, centroid_y = region['centroid']
            distance = np.sqrt((current_x - centroid_x)**2 + (current_y - centroid_y)**2)
            # You can adjust these factors to balance the importance of weight and distance
            weight_factor = region['weight'] * 0.01
            distance_factor = 1 / (1 + distance) * 1
            return weight_factor * distance_factor, region['num_points']

        unsearched_regions = [r for r in regions if r['index'] not in self.searched_regions]
        unsearched_regions.sort(key=region_priority, reverse=True)
        return unsearched_regions[0] if unsearched_regions else None

    def get_action_within_region(self, x, y, search_distribution, fov_dict):
        unvisited_points = [p for p in self.current_region['points'] if (tuple(p) not in self.visited_cells)]
        unvisited_points.sort(key=lambda p: -search_distribution[p[0], p[1]])
        return self.get_action_to_navigate_to_cell(x, y, unvisited_points[0])

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
