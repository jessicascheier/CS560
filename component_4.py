import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')

# 4.) MOVEMENT OF AN ARM
def interpolate_arm(start: np.ndarray, goal: np.ndarray, steps=100):
    
    theta_0start = start[0]
    theta_1start = start[1]
    theta_0end = goal[0]
    theta_1end = goal[1]

    path = []
    for step in range(steps):
        t = step / (steps-1)
        theta_0t = theta_0start + t * (theta_0end - theta_0start)
        theta_1t = theta_1start + t * (theta_1end - theta_1start)
        pose = (theta_0t, theta_1t)
        path.append(pose)

    # return the path
    return path

# # testing function
# path_test = interpolate_arm(np.array([0, 0]), np.array([math.pi, math.pi/2]), 5)
# print("path_test",path_test)

def forward_propagate_arm(start_pose, plan):
    # start pose describes the initial orientations of the joints
    theta_j1, theta_j2 = start_pose

    path = []
    path.append((theta_j1,theta_j2)) # appending start pose to path

    for (v1,v2,duration) in plan:
        theta_j1 += v1*duration
        theta_j2 += v2*duration
        path.append((theta_j1,theta_j2))

    return path


# def plot_arm_path(path):
#     # lengths of the links
#     l1 = 2
#     l2 = 1.5

#     # INTERPOLATE ARM
#     for pose in path:
#         theta0, theta1 = pose
#         # first joint
#         T_01 = np.array([
#             [np.cos(theta0), -np.sin(theta0), 0],
#             [np.sin(theta0), np.cos(theta0), 0],
#             [0, 0, 1]
#         ])
#         # first joint to center of the first link
#         T_12 = np.array([
#             [1, 0, l1/2],
#             [0, 1, 0],
#             [0, 0, 1]
#         ])
#         # center of first link to second joint
#         T_23 = np.array([
#             [1, 0, l1/2],
#             [0, 1, 0],
#             [0, 0, 1]
#         ])
#         # second joint to center of second link (rotate, then translate)
#         T_34 = np.array([
#             [np.cos(theta1), -np.sin(theta1), 0],
#             [np.sin(theta1), np.cos(theta1), 0],
#             [0, 0, 1]
#         ])
#         T_34b = np.array([
#             [1, 0, l2/2],
#             [0, 1, 0],
#             [0, 0, 1]
#         ])
#         # center of second link to end effector
#         T_45 = np.array([
#             [1, 0, l2/2],
#             [0, 1, 0],
#             [0, 0, 1]
#         ])

#         # combining the transformation matrices to get the positions relative to the base
#         T_L1 = T_01 @ T_12 # center of first link
#         T_J2 = T_L1 @ T_23 # second joint
#         T_L2 = T_J2 @ T_34 @ T_34b
#         T_EE = T_L2 @ T_45

#         # extracting the positions of each joint and link from the matrices
#         j1_pos = T_01[:2, 2]
#         l1_pos = T_L1[:2, 2]
#         j2_pos = T_J2[:2, 2]
#         l2_pos = T_L2[:2, 2]
#         ee_pos = T_EE[:2, 2]

#         # Plotting
#         plt.plot([0, j1_pos[0], l1_pos[0], j2_pos[0], l2_pos[0], ee_pos[0]], 
#                  [0, j1_pos[1], l1_pos[1], j2_pos[1], l2_pos[1], ee_pos[1]], 
#                  '-o')
        
#     plt.xlim(-10, 10)
#     plt.ylim(-10, 10)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Robot Arm Path')
#     plt.grid(True)
#     plt.show()
#     plt.savefig("arm_path.png")

def visualize_arm_path(path):
    # lengths of the links
    l1 = 2
    l2 = 1.5

    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Arm Animation')
    ax.grid(True)

    line, = ax.plot([], [], '-o', lw=2)
    joint1, = ax.plot([], [], 'bo')
    joint2, = ax.plot([], [], 'go')
    end_effector, = ax.plot([], [], 'ro')

    def init():
        #print("init accessed!")
        line.set_data([], [])
        joint1.set_data([], [])
        joint2.set_data([], [])
        end_effector.set_data([], [])
        return line, joint1, joint2, end_effector
    
    def update(frame):
        try:
            #print(f"update accessed! Frame: {frame}")
            theta0, theta1 = path[frame]
            T_01 = np.array([
                [np.cos(theta0), -np.sin(theta0), 0],
                [np.sin(theta0), np.cos(theta0), 0],
                [0, 0, 1]
            ])
            # first joint to center of the first link
            T_12 = np.array([
                [1, 0, l1/2],
                [0, 1, 0],
                [0, 0, 1]
            ])
            # center of first link to second joint
            T_23 = np.array([
                [1, 0, l1/2],
                [0, 1, 0],
                [0, 0, 1]
            ])
            # second joint to center of second link (rotate, then translate)
            T_34 = np.array([
                [np.cos(theta1), -np.sin(theta1), 0],
                [np.sin(theta1), np.cos(theta1), 0],
                [0, 0, 1]
            ])
            T_34b = np.array([
                [1, 0, l2/2],
                [0, 1, 0],
                [0, 0, 1]
            ])
            # center of second link to end effector
            T_45 = np.array([
                [1, 0, l2/2],
                [0, 1, 0],
                [0, 0, 1]
            ])

            # combining the transformation matrices to get the positions relative to the base
            T_L1 = T_01 @ T_12 # center of first link
            T_J2 = T_L1 @ T_23 # second joint
            T_L2 = T_J2 @ T_34 @ T_34b
            T_EE = T_L2 @ T_45

            # extracting the positions of each joint and link from the matrices
            j1_pos = T_01[:2, 2]
            l1_pos = T_L1[:2, 2]
            j2_pos = T_J2[:2, 2]
            l2_pos = T_L2[:2, 2]
            ee_pos = T_EE[:2, 2]

            line.set_data([0, j1_pos[0], j2_pos[0], ee_pos[0]],
                  [0, j1_pos[1], j2_pos[1], ee_pos[1]])

            joint1.set_data(j1_pos[0], j1_pos[1])
            joint2.set_data(j2_pos[0], j2_pos[1])
            end_effector.set_data(ee_pos[0], ee_pos[1])

            return line, joint1, joint2, end_effector
        except Exception as e:
            print(f"error in update function at frame {frame}: {e}")
    
    anim = FuncAnimation(fig, update, frames=len(path), init_func=init,
                     blit=False, interval=10, repeat=False)
    return anim

# visualizing interpolate_arm
# path = interpolate_arm(np.array([0, 0]), np.array([1*math.pi, 1/2*math.pi]), 100)
# print(path)
# anim = visualize_arm_path(path)
# anim.save('component_4a.gif', writer='imagemagick', fps=30)

# visualizing forward_propagate_arm

# plan = [(0.1, 0.1, 0.1), 
#         (0.1, 0.1, 0.1), 
#         (0.1, 0.1, 0.1), 
#         (0.1, 0.1, 0.1), 
#         (0.1, 0.1, 0.1), 
#         (0.1, 0.1, 0.1), 
#         (0.1, 2, 0.1), 
#         (0.1, 2, 0.1), 
#         (0.1, 2, 0.1), 
#         (0.1, 2, 0.2), 
#         (0.1, 2, 0.2), 
#         (0.1, 2, 0.2)]

# plan = [          
#         (0, math.pi/4, 1), 
#         (0, math.pi/4, 1), 
#         (0, math.pi/4, 1), 
#         (0, 0, 1), 
#         (0, -math.pi/4, 1), 
#         (0, -math.pi/4, 1), 
#         (0, -math.pi/4, 1),
#         (0, 0, 1)]


# plan = [          
#         (math.pi/4, 0, 1), 
#         (math.pi/4, 0, 1), 
#         (math.pi/4, 0, 1), 
#         (0, -math.pi/4, 1), 
#         (0, -math.pi/4, 1), 
#         (0, -math.pi/4, 1),
#         (-math.pi/4, math.pi/4, 1), 
#         (-math.pi/4, math.pi/4, 1), 
#         (-math.pi/4, math.pi/4, 1)]

# start_pose = (0,0)
# path = forward_propagate_arm(start_pose, plan)
#print("forward prop path:",path)
# anim = visualize_arm_path(path)
# anim.save('component_4b.gif', writer='imagemagick', fps=4)
