import time
import numpy as np
import matplotlib.pyplot as plt
from simulator import Simulator, centerline
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar
import casadi as ca


sim = Simulator()
#create triangulation and midpoints
sim.del_triangulation()
midpoints, left_edge, right_edge = sim.create_midpoints()

start_time = time.time()
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

# Lists to record velocity and acceleration commands for plotting
recorded_velocity = []
recorded_acceleration_commands = []
recorded_reference_velocity = []

# Lists to record MPC states for plotting
recorded_heading_error = []
recorded_lateral_error = []

ARC_LEN = sim.arc_length
CAR_SHAPE = sim.car_vertices


def get_path_info(total_path_info, s):
    # Find the segment corresponding to the current s value
    segment = int(s * (len(total_path_info) - 1))
    xr, yr, dx, dy, ddx, ddy = total_path_info[segment]

    return xr, yr, dx, dy, ddx, ddy

def compute_velocity_profiles(sim, total_path_info, num_points=10000 ,  initial_velocity=0.0):
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
        _, _, dx, dy, ddx, ddy = get_path_info(total_path_info, s)
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
    v_brake[-1] = v_accel[-1]  # start from end velocity
    for i in range(num_points-2, -1, -1):
        v_brake[i] = min(
            np.sqrt(v_brake[i+1]**2 + 2*abs(WHEEL_ACCEL_MIN)*ds),
            v_corner[i]
        )

    # --- Final profile: min of all three ---
    for i in range(num_points):
        v_desired[i] = min(v_corner[i], v_accel[i], v_brake[i])


    return [s_vals, v_corner, v_accel, v_brake, v_desired, arc_length_positions]

def compute_path_profile(sim, num_points = 10000):
    s_vals = np.linspace(0, 1, num_points)
    total_path_info = []

    for s in s_vals:
        xr, yr = splev(s, sim.tck)

        # First derivative (tangent)
        dx, dy = splev(s, sim.tck, der=1)

        # Second derivative (curvature-related)
        ddx, ddy = splev(s, sim.tck, der=2)

        total_path_info.append((xr, yr, dx, dy, ddx, ddy))
    
    return total_path_info


#create velocity profile for entire track
path_profile = compute_path_profile(sim)
velocity_profiles = compute_velocity_profiles(sim, path_profile)


# MPC throttling parameters
MPC_UPDATE_PERIOD = 0.05  # run MPC every 50 ms (20 Hz)
mpc_timer = 0.0          # elapsed sim time since last MPC solve
last_mpc_rate = 0.0      # last steering rate chosen by MPC


#PID
v_integral_error = 0.0
v_previous_error = 0.0

# PID gains (tune!)
v_Kp = 20.0
v_Ki = 0.35
v_Kd = 0.001

dt = 0.01  # seconds per simulation step






#build mpc for lateral control 
"""optimization variables : ey = lateral deviation from path, psi = heading error 

                           using theta = steering angle, d_theta = steering rate"""

def build_lateral_mpc (N=100, dt=0.01, L=WHEELBASE,
                      theta_min=WHEEL_ANG_MIN, theta_max=WHEEL_ANG_MAX,
                      dtheta_min=STEERING_RATE_MIN, dtheta_max=STEERING_RATE_MAX, use_slip=True):

    #3 state variables 1 control input
    nx, nu = 3, 1  # [e_y, e_psi, theta], [d_theta]

    #predict state and control over N timesteps, state at next timestep and u at current timestep
    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)

    #initial state, velocity over the horizon, path curvature over the horizon
    x0   = ca.SX.sym('x0', nx)
    vseq = ca.SX.sym('v',  N)
    kap  = ca.SX.sym('kap',N)

    # cost weights (tune later)
    w_ey, w_psi, w_theta = 2.0, 9.0, 0.01

    J = 0
    g = []; lbg = []; ubg = []

    # initial condition
    g += [X[:,0] - x0]; lbg += [0,0,0]; ubg += [0,0,0]

    for k in range(N):
        ey, psi, theta = X[0,k], X[1,k], X[2,k]
        dtheta = U[0,k]
        v_k= vseq[k]
        kap_k  = kap[k]

        v_safe = ca.fmax(v_k, 1e-3)
        w = (v_safe/L) * ca.tan(theta)
        B = ca.atan((L/2)/L * ca.tan(theta)) - (L/2)/v_safe * w

        vx_k = v_safe * ca.cos(B)
        vy_k = v_safe * ca.sin(B)

        denom_safe = ca.fmax(1 - ey * kap_k, 1e-3)
        s_dot = (vx_k * ca.cos(psi) - vy_k * ca.sin(psi)) / denom_safe



        ey_N = ey + dt  * (vy_k * ca.cos(psi) + vx_k * ca.sin(psi))
        psi_N = psi + dt * (w - s_dot * kap_k)
        theta_N = theta + dt * dtheta


        #constraint 
        g += [X[:,k+1] - ca.vertcat(ey_N, psi_N, theta_N)]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        #stage cost 
        J += w_ey*ey**2 + w_psi*psi**2 + w_theta*theta**2 

    #terminal cost
    eyN, psiN, thetaN = X[0,N], X[1,N], X[2,N]
    J += w_ey*eyN*eyN + w_psi*psiN*psiN + w_theta*thetaN*thetaN

 
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    #hard constraints
    lbx = []; ubx = []
    # States bounds: only theta is bounded; (ey, epsi) unbounded here
    for k in range(N+1):
        lbx += [-ca.inf, -ca.inf, theta_min]
        ubx += [ ca.inf,  ca.inf, theta_max]
    #dtheta bounds
    for k in range(N):
        lbx += [dtheta_min]
        ubx += [dtheta_max]

    # Pack constraints and parameters
    g_all = ca.vertcat(*g)
    p_all = ca.vertcat(x0, vseq, kap)

    # Build solver
    nlp = {'x': opt_vars, 'f': J, 'g': g_all, 'p': p_all}
    solver = ca.nlpsol(
        'mpc_lat', 'ipopt', nlp,
        {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.sb': 'yes'
        }
    )

    # Helpers for runtime use
    nxTot = nx*(N+1); nuTot = nu*N

    def pack_params(x0_val, vseq_val, kap_val):
        """
        x0_val: shape (3,), [ey, epsi, theta]
        vseq_val: shape (N,)
        kap_val:  shape (N,)
        """
        return np.concatenate([np.asarray(x0_val).ravel(),
                               np.asarray(vseq_val).ravel(),
                               np.asarray(kap_val).ravel()]).reshape(-1,1)

    def unpack_solution(sol):
        w = np.array(sol['x']).reshape(-1)
        X_opt = w[:nxTot].reshape(nx, N+1)
        U_opt = w[nxTot:].reshape(nu, N)
        return X_opt, U_opt

    meta = {
        'nx': nx, 'nu': nu, 'N': N, 'dt': dt,
        'nxTot': nxTot, 'nuTot': nuTot,
        'var_shape': (nx, N+1, nu, N),
        'use_slip': use_slip
    }

    return solver, lbx, ubx, lbg, ubg, pack_params, unpack_solution, meta

#build mpc 
mpc_solver, lbx, ubx, lbg, ubg, pack_params, unpack_solution, mpc_meta = build_lateral_mpc()
last_mpc_solution = None

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
    global sim, current_path_s, current_path_point, recorded_path_s, recorded_timestamps, past_s, ARC_LEN, CAR_SHAPE, recorded_car_x, recorded_car_y, velocity_profiles, recorded_velocity, recorded_acceleration_commands, recorded_reference_velocity, recorded_heading_error, recorded_lateral_error, mpc_timer, last_mpc_rate, path_profile
    start_time = time.perf_counter()  # Higher precision timer


    
    # EXTRACT STATE VARIABLES
    [x, y, heading, velocity, steering_angle] = x
    #[x, y, absolute heading (rad from positive x axis counterclockwise), velocity (m/s), steering angle (rad from car centerline)]
    

        
    #this should be used for everything else but the first iteration

    # Find closest point on path
    # a = past_s - PATH_CHECK_BACK/ARC_LEN
    a = past_s - PATH_CHECK_BACK/ARC_LEN
    b = past_s + PATH_CHECK_FOW/ARC_LEN 


    if a < 0:
        # split into [0, b] and [1+a, 1]
        u1, dist1 = sim.closest_point_on_spline(x, y, sim.tck, 0.0, b)
        u2, dist2 = sim.closest_point_on_spline(x, y, sim.tck, 1.0+a, 1.0)
        if dist1 < dist2:
            current_path_s = u1 
        else:
            current_path_s = u2
    elif b > 1:
        # split into [a, 1] and [0, b-1]
        u1, dist1 = sim.closest_point_on_spline(x, y, sim.tck, a, 1.0)
        u2, dist2 = sim.closest_point_on_spline(x, y, sim.tck, 0.0, b-1.0)
        
        if dist1 < dist2:
            current_path_s = u1
        else: 
            current_path_s = u2
            if b-1.0 > .9*(PATH_CHECK_FOW/ARC_LEN):
                velocity_profiles = compute_velocity_profiles(sim, path_profile, initial_velocity=velocity)
                

    else:
        # normal case
        current_path_s, _ = sim.closest_point_on_spline(x, y, sim.tck, a, b)

    # Calculate SIGNED lateral deviation
    xr, yr, dx, dy, ddx, ddy = get_path_info(path_profile, current_path_s)
    
    # Vector from path point to car
    car_to_path = np.array([x - xr, y - yr])
    
    # Path tangent vector (normalized)
    path_tangent = np.array([dx, dy])
    path_tangent = path_tangent / np.linalg.norm(path_tangent)
    
    # Path normal vector (90 degrees counterclockwise from tangent)
    path_normal = np.array([-path_tangent[1], path_tangent[0]])
    
    # Signed lateral deviation (positive = left of path, negative = right of path)
    lat_dev = np.dot(car_to_path, path_normal)

    current_path_point = [xr, yr]
    
    

    current_curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

    current_path_heading = np.arctan2(dy, dx)

    #VELOCITY PID CONTROLLER
    global v_integral_error, v_previous_error, v_Kp, v_Ki, v_Kd, dt

    #we have to first pick the correct velocity to compare to (there are 10000 points in the v_desired profile)
    #Convert continuous s (0-1) to discrete array index (0-9999)
    s_index = int(current_path_s * (len(velocity_profiles[4]) - 1))
    s_index = np.clip(s_index, 0, len(velocity_profiles[4]) - 1)  # Ensure valid index
    
    # Get reference velocity from pre-computed profile
    v_reference = velocity_profiles[4][s_index]  # velocity_profiles[4] is v_desired array
    
    # Calculate velocity error for PID
    v_error = v_reference - velocity
    v_integral_error += v_error * dt
    v_derivative_error = (v_error - v_previous_error) / dt 

    v_previous_error = v_error

    #create feasible acceleration command 
    accel_control = np.clip(v_Kp * v_error +    v_Ki * v_integral_error +   v_Kd * v_derivative_error, WHEEL_ACCEL_MIN, WHEEL_ACCEL_MAX)
    
    # Calculate MPC state variables for recording (always compute these)
    e_y   = lat_dev
    epsi = (heading - current_path_heading + np.pi) % (2*np.pi) - np.pi   # vehicle heading - path heading
    
    # always compute velocity PID (cheap)
    # only run MPC every MPC_UPDATE_PERIOD
    mpc_timer += dt
    
    if mpc_timer >= MPC_UPDATE_PERIOD:
        mpc_timer = 0.0  # reset timer
        
        # --- build and solve MPC as you currently do ---
        theta = steering_angle
        x0 = [e_y, epsi, theta]

        # Build horizon preview
        N = mpc_meta['N']

        # s_preview = [ (current_path_s + i*ds) % 1.0 for i in range(N) ]
        s_val = current_path_s
        kap  = []
        vseq = []

        for index in range(N): 
            #for each run, we have to find the reference velocity, append the velocity to the vseq array, and predict the next s value given
            # that velocity by projecting it accross dt seconds 
            vref = velocity_profiles[4][s_index]
            vseq.append(vref) 

            #use the s_val to follow indexing req for the velocity and apply to cuvature to the values are aligned
            _, _, dx, dy, ddx, ddy = get_path_info(path_profile, s_val)
            kap.append((dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5)
        
            # 0-1 
            ds = (vref * dt) / ARC_LEN

            
            s_val = (s_val + ds) % 1.0

            s_index = int(s_val * (len(velocity_profiles[4]) - 1))
    

        # Pack parameters
        p = pack_params(x0, vseq, kap)

        # Warm-start if previous solution exists
        if last_mpc_solution is not None:
            sol = mpc_solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p, x0=last_mpc_solution)
        else:
            sol = mpc_solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

        X_opt, U_opt = unpack_solution(sol)
        last_mpc_rate = float(U_opt[0,0])   # take first control input
    
    # return last known steering rate + current accel
    steering_rate = last_mpc_rate

    # Record current path_s value, car position, velocity, acceleration, and timestamp for plotting
    recorded_path_s.append(current_path_s)
    recorded_car_x.append(x)
    recorded_car_y.append(y)
    recorded_velocity.append(velocity)
    recorded_acceleration_commands.append(accel_control)
    recorded_reference_velocity.append(v_reference)
    recorded_heading_error.append(epsi)
    recorded_lateral_error.append(e_y)
    # Use simulation time (0.01 second timesteps) instead of wall-clock time
    recorded_timestamps.append(len(recorded_timestamps) * 0.01)
    computation_time = time.perf_counter() - start_time  # Higher precision
    print("current controller time: " + f"{computation_time:.6f}" + "    s value: " + f"{current_path_s:.6f}")
    past_s = current_path_s

      # Show microsecond precision
    return np.array([accel_control, steering_rate])


sim.set_controller(controller)
sim.run(20)
print(time.time() - start_time)
sim.animate()
sim.plot()

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

def plot_velocity_and_mpc_analysis():
    """Plot velocity tracking, heading error, and lateral error over time."""
    
    if len(recorded_timestamps) == 0:
        print("No data recorded!")
        return
    
    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Velocity comparison (recorded vs reference)
    ax1.plot(recorded_timestamps, recorded_velocity, 'b-', linewidth=2, label='Actual Velocity', alpha=0.8)
    ax1.plot(recorded_timestamps, recorded_reference_velocity, 'r--', linewidth=2, label='Reference Velocity', alpha=0.8)
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity Tracking Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Heading error over time
    ax2.plot(recorded_timestamps, recorded_heading_error, 'r-', linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Heading Error (rad)')
    ax2.set_title('Heading Error Over Time (Vehicle - Path)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lateral error over time
    ax3.plot(recorded_timestamps, recorded_lateral_error, 'orange', linewidth=2, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Lateral Error (m)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Lateral Error Over Time (Distance from Path)')
    ax3.grid(True, alpha=0.3)
    
    # Add some statistics
    if len(recorded_velocity) > 0:
        avg_velocity = np.mean(recorded_velocity)
        max_velocity = np.max(recorded_velocity)
        
        # Calculate RMS errors
        rms_heading_error = np.sqrt(np.mean([e**2 for e in recorded_heading_error]))
        rms_lateral_error = np.sqrt(np.mean([e**2 for e in recorded_lateral_error]))
        
        stats_text = f"""
        Performance Statistics:
        • Avg velocity: {avg_velocity:.2f} m/s
        • Max velocity: {max_velocity:.2f} m/s
        • RMS heading error: {rms_heading_error:.3f} rad
        • RMS lateral error: {rms_lateral_error:.3f} m
        • Final heading error: {recorded_heading_error[-1]:.3f} rad
        • Final lateral error: {recorded_lateral_error[-1]:.3f} m
        """
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_velocity_vs_position():
    """Plot velocity profiles as function of track position instead of time."""
    
    if len(recorded_path_s) == 0:
        print("No data recorded!")
        return
        
    # Convert path_s to arc length positions
    recorded_arc_positions = [s * ARC_LEN for s in recorded_path_s]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Velocity vs position
    ax1.scatter(recorded_arc_positions, recorded_velocity, c=recorded_timestamps, 
                s=20, alpha=0.7, cmap='viridis', label='Actual Velocity')
    ax1.plot(velocity_profiles[5], velocity_profiles[4], 'r-', 
             linewidth=3, alpha=0.8, label='Optimal Velocity Profile')
    
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Time (seconds)', rotation=270, labelpad=15)
    
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity vs Track Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Acceleration vs position  
    ax2.scatter(recorded_arc_positions, recorded_acceleration_commands, c=recorded_timestamps,
                s=20, alpha=0.7, cmap='plasma')
    ax2.axhline(y=WHEEL_ACCEL_MAX, color='r', linestyle=':', alpha=0.7)
    ax2.axhline(y=WHEEL_ACCEL_MIN, color='r', linestyle=':', alpha=0.7)
    
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Time (seconds)', rotation=270, labelpad=15)
    
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_xlabel('Arc Length Position (m)')
    ax2.set_title('Acceleration Commands vs Track Position')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_vel_profiles(velocity_profiles):
    s_vals, v_corner, v_accel, v_brake, v_desired, arc_length_positions = velocity_profiles

    # --- Plot with arc length instead of point indices ---
    plt.figure(figsize=(12,6))
    plt.plot(arc_length_positions, v_brake, 'r-', label="Braking Limit", linewidth=2, alpha=0.8)
    plt.plot(arc_length_positions, v_accel, 'y-', label="Accelerating Limit", linewidth=2, alpha=0.8)
    plt.plot(arc_length_positions, v_desired, 'b-', label="Final Velocity Profile", linewidth=3)
    plt.xlabel("Arc Length [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title(f"Velocity Profile vs Arc Length (Total track length: {ARC_LEN:.2f} m)")
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
    • Track length: {ARC_LEN:.2f} m
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.show()

# plot_vel_profiles(velocity_profiles)

# Plot analysis of velocity and MPC performance
plot_velocity_and_mpc_analysis()
plot_velocity_vs_position()


