import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 3.) RIGID BODY IN MOTION

def interpolate_rigid_body(start_pose, goal_pose):
    # Polynomial Time-Scaling
    # Chapter 9 - Trajectory Generation (see pgs. 329-330)
    # https://youtu.be/sWPpq9-5YOc?si=l8IWILXuTBrmv0c7
    x0, y0, theta0 = start_pose
    xG, yG, thetaG = goal_pose
    t = np.linspace(0, 1, 100)

    x_a0 = x0
    x_a1 = 0
    x_a2 = 3 * (xG - x0)
    x_a3 = 2 * (x0 - xG)
    x = x_a0 + x_a1*t +  x_a2*t**2 + x_a3*t**3

    y_a0 = y0
    y_a1 = 0
    y_a2 = 3 * (yG - y0)
    y_a3 = 2 * (y0 - yG)
    y = y_a0 + y_a1*t +  y_a2*t**2 + y_a3*t**3
    
    theta_diff = np.arctan2(np.sin(thetaG - theta0), np.cos(thetaG - theta0))
    thetaG = theta0 + theta_diff
    theta_a0 = theta0
    theta_a1 = 0
    theta_a2 = 3 * (thetaG - theta0)
    theta_a3 = 2 * (theta0 - thetaG)
    theta = theta_a0 + theta_a1*t +  theta_a2*t**2 + theta_a3*t**3
    
    path = np.vstack((x, y, theta)).T
    return path

def forward_propagate_rigid_body(start_pose, plan):
    # Chapter 3 - Rigid-Body Motions (see pgs. 64-65)
    x, y, theta = start_pose
    path = [start_pose]

    for velocity, duration in plan:
        Vx, Vy, Vtheta = velocity
        dt = 0.1
        for t in np.arange(0, duration, dt):
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            vector = np.array([Vx, Vy])
            Rv = R @ vector

            dx, dy = Rv * dt
            x += dx
            y += dy
            theta += Vtheta * dt
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            path.append((x, y, theta))

    return np.array(path)

def visualize_path(path):
    # https://matplotlib.org/stable/users/explain/animations/animations.html
    fig, ax = plt.subplots()
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])

    x_data = path[:,0]
    y_data = path[:,1]
    theta_data = path[:,2]

    ax.plot(path[:,0], path[:,1], 'b--')
    robot_body, = ax.plot([], [], 'r-')

    def initialize():
        robot_body.set_data([], [])
        return robot_body

    def animate(frame):
        x = x_data[frame]
        y = y_data[frame]
        theta = theta_data[frame]
        length, width = 0.5, 0.3
        
        robot = np.array([
            [length/2, width/2],
            [length/2, -width/2],
            [-length/2, -width/2],
            [-length/2, width/2],
            [length/2, width/2]
        ]).T
        
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        rotated_robot = (R @ robot)
        translated_robot = rotated_robot + np.array([[x], [y]])
        robot_body.set_data(translated_robot[0,:], translated_robot[1,:])
        
        return robot_body

    ani = FuncAnimation(fig, animate, frames=len(path), 
                        init_func=initialize, blit=False, interval=50)

    plt.show()

# Test case for interpolate_rigid_body():
# start_pose = (0, 0, 0)
# goal_pose = (6, 3, np.pi/2)
# path = interpolate_rigid_body(start_pose, goal_pose)

# Test case for forward_propagate_rigid_body():
# plan = [((1.0, 0.25, 0.1), 5), ((0.5, 0.25, 1.0), 3)]
# propagated_path = forward_propagate_rigid_body(start_pose, plan)

# Visualize both paths
# visualize_path(propagated_path)
# visualize_path(path)
