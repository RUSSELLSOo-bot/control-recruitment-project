import time
import numpy as np
import matplotlib.pyplot as plt
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
past_s = 0.0

# Lists to record path_s values over time for plotting
recorded_path_s = []
recorded_timestamps = []

# Lists to record car position over time for plotting
recorded_car_x = []
recorded_car_y = []

ARC_LEN = sim.arc_length
CAR_SHAPE = sim.car_vertices

def get_path_info(sim, s):
    xr, yr = splev(s, sim.tck)

    # First derivative (tangent)
    dx, dy = splev(s, sim.tck, der=1)

    # Second derivative (curvature-related)
    ddx, ddy = splev(s, sim.tck, der=2)

    return xr, yr, dx, dy, ddx, ddy


start_time = time.time()


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
    global sim, current_path_s, current_path_point, recorded_path_s, recorded_timestamps, past_s, ARC_LEN, CAR_SHAPE, recorded_car_x, recorded_car_y
    
    
    
    # Constants are automatically accessible (defined outside function)
    # WHEELBASE, WHEEL_ANG_MAX, PATH_CHECK_FOW, etc. can be used directly
    
    # EXTRACT STATE VARIABLES
    [x, y, heading, velocity, steering_angle] = x
    

        
    #this should be used for everything else but the first iteration

    # Find closest point on path
    # a = past_s - PATH_CHECK_BACK/ARC_LEN
    a = past_s - PATH_CHECK_BACK/ARC_LEN
    b = past_s + PATH_CHECK_FOW/ARC_LEN 


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

    xr, yr, dx, dy, ddx, ddy = get_path_info(sim, current_path_s)
    current_path_point = [xr, yr]
    

    current_curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

    """Now we want to make a velocity controller separate based on the total curvature of the path"""
    # this is an absolute constraint
    v_corner = np.min([np.sqrt(MAX_TOTAL_ACCEL / abs(current_curvature)), 20])

    #this is a constraint with a foward sweep, from starting to finish what is feasible for our car per point to accerlate to 
    a_accel_feasible = np.min([MAX_TOTAL_ACCEL, current_curvature*velocity**2])
    a_accel_max = np.abs(WHEEL_ACCEL_MIN) * np.sqrt(1 - (a_accel_feasible/MAX_TOTAL_ACCEL)**2)
    v_accel = np.sqrt(velocity**2 + 2*a_accel_max((current_path_s - past_s)*ARC_LEN))

    # this is a constraint with a backward sweep, from finish to start what is feasible for our car to decelerate to
    a_decel_feasible = np.min(MAX_TOTAL_ACCEL, current_curvature*velocity**2)
    a_brake_max = np.abs(WHEEL_ACCEL_MIN) * np.sqrt(1 - (a_decel_feasible/MAX_TOTAL_ACCEL)**2 )
    v_decel = np.sqrt(velocity**2 + 2*a_brake_max((past_s - current_path_s)*ARC_LEN))

    v_desired = np.min([v_corner, v_accel, v_decel])



    # COMPUTE LATERAL DEVIATION
    # RULES FOR CLIPPING
    acceleration = np.clip(10, WHEEL_ACCEL_MIN, WHEEL_ACCEL_MAX)
    steering_rate = np.clip(0, STEERING_RATE_MIN, STEERING_RATE_MAX)
    
    # Record current path_s value, car position, and timestamp for plotting
    # Record current time for plotting (approximate simulation time)
    #current_time = len(recorded_timestamps) * 0.01  # 0.01 second timesteps
    # recorded_path_s.append(current_path_s)
    # recorded_car_x.append(x)
    # recorded_car_y.append(y)
    # recorded_timestamps.append(time.time() - start_time)
    past_s = current_path_s
    
    return np.array([acceleration, steering_rate])


# sim.set_controller(controller)
# sim.run()
# sim.animate()
# sim.plot()

def plot_path_s_and_position_over_time():
    """Plot the recorded path_s values and car position as a function of time."""
    
    if len(recorded_path_s) == 0:
        print("No data recorded!")
        return
    
    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: s values (0-1) as function of time
    ax1.plot(recorded_timestamps, recorded_path_s, 'b-', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Path Parameter s (0-1)')
    ax1.set_title('Path Parameter s Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Car X position over time
    ax2.plot(recorded_timestamps, recorded_car_x, 'r-', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Car X Position (m)')
    ax2.set_title('Car X Position Over Time')
    ax2.grid(True, alpha=0.3)


    # Plot 3: Car Y position over time
    ax3.plot(recorded_timestamps, recorded_car_y, 'g-', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Car Y Position (m)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Car Y Position Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Add some statistics
    if len(recorded_timestamps) > 0 and len(recorded_path_s) > 0:
        total_time = recorded_timestamps[-1]
        final_s = recorded_path_s[-1]
        initial_s = recorded_path_s[0]
        s_progress = final_s - initial_s
        
        # Handle wrap-around case (if s goes from near 1 back to near 0)
        if s_progress < -0.5:  # Wrapped around
            s_progress += 1.0
        
        avg_s_rate = s_progress / total_time if total_time > 0 else 0
        
        # Calculate distance traveled
        total_distance = 0
        if len(recorded_car_x) > 1:
            for i in range(1, len(recorded_car_x)):
                dx = recorded_car_x[i] - recorded_car_x[i-1]
                dy = recorded_car_y[i] - recorded_car_y[i-1]
                total_distance += np.sqrt(dx**2 + dy**2)
        
        avg_speed = total_distance / total_time if total_time > 0 else 0
        
        stats_text = f"""
        Statistics:
        • Total simulation time: {total_time:.2f} s
        • Initial s value: {initial_s:.3f}
        • Final s value: {final_s:.3f}
        • s progress: {s_progress:.3f}
        • Average s rate: {avg_s_rate:.3f} s⁻¹
        • Laps completed: {s_progress:.2f}
        • Distance traveled: {total_distance:.2f} m
        • Average speed: {avg_speed:.2f} m/s
        • Final position: ({recorded_car_x[-1]:.2f}, {recorded_car_y[-1]:.2f}) m
        """
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def compute_and_plot_velocity_profiles(sim, num_points=10000):
    # Discretize spline
    s_vals = np.linspace(0, 1, num_points)
    arc_len = sim.arc_length
    ds = arc_len / (num_points - 1)
    
    # Create arc length array for plotting (in meters)
    arc_length_positions = s_vals * arc_len

    # Arrays
    v_corner = np.zeros(num_points)
    v_accel  = np.zeros(num_points)
    v_brake  = np.zeros(num_points)
    v_desired  = np.zeros(num_points)

    #cornering limit case, find the max velocity at all discretized points on the path that satisfy the cornering acceraltion constraint
    for i, s in enumerate(s_vals):
        _, _, dx, dy, ddx, ddy = get_path_info(sim, s)
        curvature = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

        #edge case doesnt matter it will pick the smallest value anyway
        v_corner[i] = np.sqrt(MAX_TOTAL_ACCEL / abs(curvature))

    #forward sweep, having foresight to see maximum acceration feasible from the start, still constrained by teh cornering limit
    v_accel[0] = 0.0
    for i in range(num_points-1):
        v_accel[i+1] = min(
            np.sqrt(v_accel[i]**2 + 2*WHEEL_ACCEL_MAX*ds),
            v_corner[i+1]
        )

    #backward sweep, having hindsight to see maximum decceleration feasible from the end, still constrained by the cornering limit
    v_brake[-1] = 0.0
    for i in range(num_points-2, -1, -1):
        v_brake[i] = min(
            np.sqrt(v_brake[i+1]**2 + 2*abs(WHEEL_ACCEL_MIN)*ds),
            v_corner[i]
        )

    # --- Final profile: min of all three ---
    for i in range(num_points):
        v_desired[i] = min(v_corner[i], v_accel[i], v_brake[i])

    # --- Plot with arc length instead of point indices ---
    plt.figure(figsize=(12,6))
    plt.plot(arc_length_positions, v_brake, 'r-', label="Braking Limit", linewidth=2, alpha=0.8)
    plt.plot(arc_length_positions, v_accel, 'y-', label="Accelerating Limit", linewidth=2, alpha=0.8)
    plt.plot(arc_length_positions, v_desired, 'b-', label="Final Velocity Profile", linewidth=3)
    plt.xlabel("Arc Length [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"Velocity Profile vs Arc Length (Total track length: {arc_len:.2f} m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    max_velocity = np.max(v_desired)
    avg_velocity = np.mean(v_desired)
    min_velocity = np.min(v_desired[v_desired > 0])  # Exclude zero velocities

    stats_text = f"""
    Statistics:
    • Max velocity: {max_velocity:.2f} m/s
    • Avg velocity: {avg_velocity:.2f} m/s
    • Min velocity: {min_velocity:.2f} m/s
    • Track length: {arc_len:.2f} m
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.show()

    return s_vals, v_corner, v_accel, v_brake, v_desired, arc_length_positions

compute_and_plot_velocity_profiles(sim)
# Plot the recorded path_s values and car position
#plot_path_s_and_position_over_time()

