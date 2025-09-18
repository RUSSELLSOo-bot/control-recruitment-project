import numpy as np
from simulator import Simulator, centerline
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar

sim = Simulator()
#create triangulation and midpoints
sim.del_triangulation()
midpoints, left_edge, right_edge = sim.create_midpoints()

sim.order_cones(midpoints)
#interpolate path, can access the path progress via spline object tck
sim.spline_path(midpoints)

# Vehicle and control constants
WHEELBASE = 1.58  # meters
WHEEL_ANG_MAX = 0.7  # radians
WHEEL_ANG_MIN = -0.7  # radians
STEERING_RATE_MAX = 1.0  # radians per second
STEERING_RATE_MIN = -1.0  # radians per second
WHEEL_ACCEL_MAX = 10.0  # meters per second squared
WHEEL_ACCEL_MIN = -4.0  # meters per second squared
MAX_TOTAL_ACCEL = 12.0  # meters per second squared
MIN_TURN_RADIUS = WHEELBASE / np.tan(WHEEL_ANG_MAX)  # meters

# Path following parameters
PATH_CHECK_FOW = 0.5  # m - forward lookahead distance
PATH_CHECK_BACK = 0.2  # m - backward check distance

# Global variable to track current path position for visualization
current_path_s = 0.0
current_path_point = [0.0, 0.0]

arc_len = sim.arc_length
car_shape = sim.car_vertices

def get_path_info(sim, s):
    xr, yr = splev(s, sim.tck)

    # First derivative (tangent)
    dx, dy = splev(s, sim.tck, der=1)

    # Second derivative (curvature-related)
    ddx, ddy = splev(s, sim.tck, der=2)

    return xr, yr, dx, dy, ddx, ddy





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
    global sim, current_path_s, current_path_point
    
    # Constants are automatically accessible (defined outside function)
    # WHEELBASE, WHEEL_ANG_MAX, PATH_CHECK_FOW, etc. can be used directly
    
    # EXTRACT STATE VARIABLES
    [x, y, heading, velocity, steering_angle] = x
    past_s = current_path_s
    #when the code is starting up, we want to initialize the pastxr and pastyr variables
    if x == 0 and y == 0:
        current_path_s = 0  # Start at beginning of path
        xr, yr, dx, dy, ddx, ddy = get_path_info(sim, current_path_s)
        current_path_point = [xr, yr]
        past_s = current_path_s
        
        
    #this should be used for everything else but the first iteration
    else:
        # Find closest point on path
        a = (past_s - PATH_CHECK_BACK)/arc_len
        b = ((past_s + PATH_CHECK_FOW)/arc_len ) % 1
        # current_path_s, lat_dev = sim.closest_point_on_spline( a, b)

        if a < 0:
            # split into [0, b] and [1+a, 1]
            u1, dist1 = sim.closest_point_on_spline(x, y, sim.tck, 0.0, b)
            u2, dist2 = sim.closest_point_on_spline(x, y, sim.tck, 1.0+a, 1.0)
            current_path_s = u1 if dist1 < dist2 else u2
        elif b > 1:
            # split into [a, 1] and [0, b-1]
            u1, dist1 = sim.closest_point_on_spline(x, y, sim.tck, a, 1.0)
            u2, dist2 = sim.closest_point_on_spline(x, y, sim.tck, 0.0, b-1.0)
            current_path_s = u1 if dist1 < dist2 else u2
        else:
            # normal case
            current_path_s, lat_dev = sim.closest_point_on_spline(x, y, sim.tck, a, b)

        current_path_s = current_path_s % 1.0

        xr, yr, dx, dy, ddx, ddy = get_path_info(sim, current_path_s)
        current_path_point = [xr, yr]
        past_s = current_path_s
    


    # COMPUTE LATERAL DEVIATION
    # RULES FOR CLIPPING
    acceleration = np.clip(10, WHEEL_ACCEL_MIN, WHEEL_ACCEL_MAX)
    steering_rate = np.clip(0, STEERING_RATE_MIN, STEERING_RATE_MAX)
    





    return np.array([acceleration, steering_rate])


sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()

# Uncomment the line below to test the closest point visualization
#test_closest_point_visualization()