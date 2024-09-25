import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

# 2.) UNIFORM RANDOM ROTATIONS

def random_rotation_matrix(naive: bool) -> np.ndarray:
    if naive:
        # create random euler angles
        alpha = random.uniform(0, 2 * math.pi)
        beta = random.uniform(0, math.pi)
        gamma = random.uniform(0, 2 * math.pi)
        # convert each euler angle into a rotation matrix
        R_x = np.array([[1,0,0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        R_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
        R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0,0,1]])
        # multiply each rotation matrix (ZYX is the order chosen) to return the final rotation matrix
        R = R_z @ R_y @ R_x
        return R
    else:
        # 3 random variables
        x1 = random.uniform(0,1)
        x2 = random.uniform(0,1)
        x3 = random.uniform(0,1)

        theta = 2 * math.pi * x1 
        phi = 2 * math.pi * x2 
        z = x3 

        # construct a vector for performing the reflection
        V = np.array([[np.cos(phi)*math.sqrt(z)], [np.sin(phi)*math.sqrt(z)], [math.sqrt(1-z)]])


        R_z = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
            ])
        M = (2*V@V.T-np.eye(3))@R_z
        return M

def rotation_visualization(naive: bool, epsilon: float=0.05, iterations=1000):
    # generating a set of vectors v1'-v0' to visualize the distribution of rotations
    vp0_set = []
    vp1_set = []
    for i in range(iterations):
        R = random_rotation_matrix(naive)

        v0 = np.array([[0],[0],[1]])
        v1 = np.array([[0], [epsilon], [0]]) + v0
        #print("v0:",v0)
        #print("v1:",v1)
        vp0 = R @ v0
        vp1 = R @ v1 # - v0???
        #print("vp0:",vp0)
        #print("vp1:",vp1)
        vp0_set.append(vp0)
        vp1_set.append(vp1)

    # plot vector from v0p to v1p
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection = '3d')

    # plotting a sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1*np.outer(np.cos(u), np.sin(v))
    y = 1*np.outer(np.sin(u), np.sin(v))
    z = 1*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.2)

    # plotting the set of relative vectors
    for i in range(len(vp0_set)):
        vector_components = vp1_set[i] - vp0_set[i]
        ax.quiver(vp0_set[i][0],vp0_set[i][1],vp0_set[i][2], vector_components[0],vector_components[1],vector_components[2], color='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title('visualizing the distribution of rotations with naive = '+str(naive))
    #print("displaying graph!")
    plt.show()
    # if naive:
    #     plt.savefig("component_4a.png")
    # else:
    #     plt.savefig("component_4b.png")
    plt.clf()

# testing matrix generation functions
# matrix = random_rotation_matrix(True)
# print(matrix)
# print("random rotation orthogonal check:", check_SOn(matrix))
# matrix = random_rotation_matrix(False)
# print(matrix)
# print("random rotation orthogonal check:", check_SOn(matrix))

# visualizing R matrices
# rotation_visualization(True)
# rotation_visualization(False)

# 2.1 extra

def random_quaternion(naive: bool) -> np.ndarray:
    if naive:
        # See page 2 of Reference 2
        # https://youtu.be/R5CpG1eq5uQ?si=Ln9-k-HzuG29j2w6
        psi = np.random.uniform(0, 2*np.pi) # yaw
        theta = np.random.uniform(0, np.pi) # pitch
        phi = np.random.uniform(0, 2*np.pi) # roll

        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        return Rz @ Ry @ Rx

    else:
        # See Algorithm 2 Pseudocode
        s = np.random.rand()
        sigma1 = np.sqrt(1-s)
        sigma2 = np.sqrt(s)
        theta1 = 2*np.pi*np.random.rand()
        theta2 = 2*np.pi*np.random.rand()

        w = np.cos(theta2)*sigma2
        x = np.sin(theta1)*sigma1
        y = np.cos(theta1)*sigma1
        z = np.sin(theta2)*sigma2

        return np.array([w,x,y,z])
