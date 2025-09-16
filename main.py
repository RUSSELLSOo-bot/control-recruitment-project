import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

def controller(x):
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]

        variable	lower bound	upper bound
            θ           -0.7	    0.7

            θ           -1.0	    1.0
            ˙
            a           -4	        10

        the wheelbase (distance between front and rear wheels) is 1.58 meters.
        the maximum acceleration the car can handle (in x and y combined) is 12 meters per second per second.
        
    """

    #set constraints
    
    wheelbase = 1.58  # meters
    wheel_ang_max = 0.7  # radians
    wheel_ang_min = -0.7  # radians
    steering_rate_max = 1.0  # radians per second
    steering_rate_min = -1.0  # radians per second
    wheel_accel_max = 10.0  # meters per second squared
    wheel_accel_min = -4.0  # meters per second squared
    max_total_accel = 12.0  # meters per second squared
    min_turn_radius = wheelbase / np.tan(wheel_ang_max)  # meters






    car_shape = sim.car_vertices
    
    # ACCESS CONE POSITIONS AND COMPUTE DESIRED PATH
    all_cones = sim.cones           # All cone positions (N, 2)
    left_cones = sim.left_cones     # Blue cones on left side  
    right_cones = sim.right_cones   # Orange cones on right side
    
    # COMPUTE DESIRED PATH (triangulated from cones)
    desired_path = sim.compute_desired_path(lookahead_distance=30.0, num_points=50)
    
    # Extract current state
    pos_x, pos_y, heading, velocity, steering_angle = x
    current_pos = np.array([pos_x, pos_y])
    
    # Find closest waypoint on desired path
    distances_to_path = np.linalg.norm(desired_path - current_pos, axis=1)
    closest_idx = np.argmin(distances_to_path)
    
    # Look ahead for target point
    lookahead_idx = min(closest_idx + 5, len(desired_path) - 1)  # Look 5 points ahead
    target_point = desired_path[lookahead_idx]
    
    # Calculate steering to reach target point
    dx = target_point[0] - pos_x
    dy = target_point[1] - pos_y
    target_heading = np.arctan2(dy, dx)
    heading_error = target_heading - heading
    
    # Normalize angle to [-π, π]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # Simple proportional control
    acceleration = 2.0  # Constant acceleration for now
    steering_rate = 0.5 * heading_error  # Proportional steering



    # RULES FOR CLIPPING
    acceleration = np.clip(acceleration, -4, 10)
    steering_rate = np.clip(steering_rate, -1, 1)
    
    return np.array([acceleration, steering_rate])

sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()