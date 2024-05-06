import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, search_area, max_iter=1000, expand_dist=0.5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.search_area = search_area
        self.max_iter = max_iter
        self.expand_dist = expand_dist
        self.node_list = []

    def plan(self):
        self.node_list.append(self.start)
        for i in range(self.max_iter):
            rnd_node = self.generate_random_node()
            nearest_node_index = self.nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_node_index]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)
            if self.check_collision(new_node):
                self.node_list.append(new_node)
            if self.calculate_distance(self.node_list[-1], self.goal) <= self.expand_dist:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dist)
                if self.check_collision(final_node):
                    return self.generate_path(len(self.node_list) - 1)
        return None

    def generate_random_node(self):
        if np.random.rand() < 0.05:
            return Node(self.goal.x, self.goal.y)
        else:
            return Node(np.random.uniform(self.search_area[0], self.search_area[1]),
                        np.random.uniform(self.search_area[2], self.search_area[3]))

    def nearest_node_index(self, rnd_node):
        distance_list = [self.calculate_distance(node, rnd_node) for node in self.node_list]
        return np.argmin(distance_list)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calculate_distance_and_angle(new_node, to_node)
        new_node.x += extend_length * np.cos(theta)
        new_node.y += extend_length * np.sin(theta)
        new_node.parent = from_node
        return new_node

    def check_collision(self, node):
        for (ox, oy, w, h) in self.obstacle_list:
            if ox <= node.x <= ox + w and oy <= node.y <= oy + h:
                return False  # collision
        return True  # safe

    def generate_path(self, goal_index):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[goal_index].parent is not None:
            node = self.node_list[goal_index]
            path.append([node.x, node.y])
            goal_index = self.node_list.index(node.parent)
        path.append([self.start.x, self.start.y])
        return path

    def calculate_distance(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return np.hypot(dx, dy)

    def calculate_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

def plot_obstacles(obstacle_list):
    for (ox, oy, w, h) in obstacle_list:
        plt.plot([ox, ox + w, ox + w, ox, ox], [oy, oy, oy + h, oy + h, oy], "red")

def main():
    # Load the image and labels
    image_directory = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\images\\'
    labels_directory = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\labels\\'

    # Define obstacle positions and sizes
    obstacle_list = [(-3, 0, 2, 5), (1, 1, 3, 2), (-5, -2, 2, 3)]

    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)

            # Plan the path
            rrt = RRT(start=(-5, -5), goal=(5, 5), obstacle_list=obstacle_list, search_area=(-10, 10, -10, 10))
            path = rrt.plan()

            # Plot the result
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if path is not None:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], "-r", label="path")
            plot_obstacles(obstacle_list)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Path Planning Result")
            plt.axis("equal")
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.grid(True)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
