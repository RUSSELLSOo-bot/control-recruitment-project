import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


_centerline = ca.external(
    'centerline', 
    ca.Importer(
        'assets/cline_func.c',
        'shell'
    )
)
centerline = np.vectorize(
    lambda x: _centerline(x).toarray().flatten(),
    signature='()->(2)',
    doc="""get track centerline at distance `x` from the start.

        Args:
            x (float or array-like): distance along centerline to get.

        Returns:
            array: shape (2,) if `x` is a float or (N, 2) if `x` is an array-like of length N.
        """
)


class Simulator:
    def __init__(self, controller_callback=None):
        """initializes a Simulator

        Args:
            controller_callback (function): function that takes in a numpy array of [x, y, theta, v, phi] and outputs a control [a, phidot]
        """
        self.dynamics = ca.external('F', ca.Importer('assets/system_dynamics.c', 'shell'))
        self.left_cones = np.load('assets/left.npy')
        self.right_cones = np.load('assets/right.npy')
        self._cones = np.concatenate([self.left_cones, self.right_cones], axis=0)
        self.car_outline = np.load('assets/pts_mat.npy')
        self.A = np.load('assets/a_mat.npy')
        self.b = np.load('assets/b_mat.npy')
        self._steering_limits = (-0.7, 0.7)
        self._accel_limits = (-10, 4)
        self._steering_vel_limits = (-1, 1)
        if controller_callback is not None:
            self.cb = controller_callback
    def set_controller(self, controller_callback):
        self.cb = controller_callback
    
    @property
    def centerline(self):
        """get track centerline function.
        
        Returns:
            function: centerline function that takes distance and returns (x, y) coordinates.
        """
        return centerline
    
    @property
    def cones(self):
        """get all cone locations.

        Returns:
            array: shape (N, 2) array of cone coordinates.
        """
        return self._cones
    @property
    def steering_limits(self):
        """get the feasible range for the front tires.

        Returns:
            (float, float): (min, max) angle of front tires, in radians.
        """
        return self._steering_limits
    @property 
    def lbu(self):
        """get the lower bound on the control matrix $u=[a, \dot{\phi}]$

        Returns:
            array: shape (2,), lower bound for [acceleration, steering velocity]
        """
        return np.array([self._accel_limits[0], self._steering_vel_limits[0]])
    @property
    def ubu(self):
        """get the upper bound on the control matrix $u=[a, \dot{\phi}]$

        Returns:
            array: shape (2,), upper bound for [acceleration, steering velocity]
        """
        return np.array([self._accel_limits[1], self._steering_vel_limits[1]])
    @property
    def car_vertices(self):
        return self.car_outline
    
    """RUSSELLLLLLLLLL CODEEEEEE"""
    class Polygon:
        
        #vertices should be an (N, 2) array of points, ORDERED LOCAITONALLY 
        def __init__(self, vertices: np.ndarray):
            self.cx = np.mean(vertices[:, 0])
            self.cy = np.mean(vertices[:, 1])

            self.points = vertices
            self.edges = self.create_edges(vertices)
            
            self.circle = []
            self.circle.append(self.create_circles_with_triangle_vert())

        def create_edges(self, vertices):
            
            # edge should be a (p1, p2) tuple
            edges = []
            for i in range(len(vertices)):
                p1 = vertices[i]
                p2 = vertices[(i + 1) % len(vertices)]
                edges.append((p1, p2))
            return edges
        
        def check_inside_polygon (self, point):
            #checks for edges that are to the right of the selected point, and the intersection of a imaginary ray to the right
            # odd intersections means inside, even means outside
            x, y = point
            intercept_count = 0
            for i in self.edges:
                p1, p2 = i
                x1, y1 = p1[0], p1[1]  
                x2, y2 = p2[0], p2[1]

                if min(y1, y2) < y <= max(y1, y2) and x <= max(x1, x2):
                    if y1 != y2:
                        xintercept = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xintercept:
                        intercept_count += 1

            inside = (intercept_count % 2 == 1)
            return inside
        
        def create_circles_with_triangle_vert(self):
            #circle should be a (x, y, r) 
            if len(self.points) == 3:
                (x1, y1), (x2, y2), (x3, y3) = self.points
                
                D = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                
                # fix for corner cases of determinant being zero
                if D == 0:
                    return None
                
                x_c = ((x1**2 + y1**2)*(y2 - y3) +
                        (x2**2 + y2**2)*(y3 - y1) +
                        (x3**2 + y3**2)*(y1 - y2)) / D

                y_c = ((x1**2 + y1**2)*(x3 - x2) +
                        (x2**2 + y2**2)*(x1 - x3) +
                        (x3**2 + y3**2)*(x2 - x1)) / D

                r = np.sqrt((x_c - x1)**2 + (y_c - y1)**2)

                return (x_c, y_c, r)
            
            else:
                pass

        def check_inside_circle(self, point):
            for i in self.circle:
                if i is None:  # handle collinear triangles
                    continue
                x_c, y_c, r = i
                x, y = point
                if (x - x_c)**2 + (y - y_c)**2 < r**2:
                    return True
            return False        

                
   
    def del_triangulation(self):
        """Create a bounding triangle that contains all points."""
        # Find bounding box of all points
        maxx, minx, maxy, miny = -np.inf, np.inf, -np.inf, np.inf
        point_count = 0
        

        for i in self.cones:
            maxx = max(maxx, i[0])
            minx = min(minx, i[0])
            maxy = max(maxy, i[1])
            miny = min(miny, i[1])
        
        # Add padding to ensure all points are inside
        width = 50 * (maxx - minx)
        height = 50 * (maxy - miny)
        padding = 0.1 * max(width, height)  
        

        triangle_vertices = np.array([
            [minx - padding, miny - padding],  # Bottom-left
            [maxx + padding, miny - padding],  # Bottom-right
            [np.mean([minx, maxx ]), maxy + padding]   # Top-left (forms right angle)
        ])
        
        start_tri = self.Polygon(triangle_vertices)
        polygon_list = [start_tri]

        while point_count < len(self.cones):
            poly_index = []
            #pick new point and see if it is in anyones circle

            for index, tri in enumerate(polygon_list):
                
                if tri.check_inside_circle(self.cones[point_count]):
                    # Split triangle into 3 new triangles
                    poly_index.append(index)
                    
            
            edges = []
            #for all circles that have the point within them, delete the polygon but keep the good edges
            for index in poly_index:
                tri = polygon_list[index]
                for e in tri.edges:
                    e_sorted = tuple(sorted(map(tuple, e)))  # normalize
                    if e_sorted in edges:
                        edges.remove(e_sorted)  # shared edge = discard
                    else:
                        edges.append(e_sorted)  # unique edge = keep
            
            # delete the polygons after the important edges are found and recorded within edges
            for index in sorted(poly_index, reverse=True):
                del polygon_list[index]


            #polygonal hole is made with saved edges and deleted polygons, connect verticies of plygonal hole with new point to make new tris
            new_point = self.cones[point_count]
            for edge in edges:
                p1, p2 = edge
                # Fix 3: Explicitly cast back to np.array to avoid mixed types
                p1 = np.array(p1)
                p2 = np.array(p2)
                new_polygon = self.Polygon(np.array([p1, p2, new_point]))
                polygon_list.append(new_polygon)

                


            point_count += 1

        del polygon_list[0]  # remove the starting triangle
        
        # Fix 1: Remove triangles containing super-triangle vertices
        super_verts = triangle_vertices.tolist()
        polygon_list = [
            t for t in polygon_list
            if not any((p.tolist() in super_verts) for p in t.points)
        ]
        
        def clean_triangulation(polygon_list, points):
            cleaned = []
            seen = set()

            for poly in polygon_list:
                if len(poly.points) != 3:
                    continue
                circle = poly.create_circles_with_triangle_vert()
                if circle is None:
                    continue
                x_c, y_c, r = circle

                # Check if any point lies strictly inside this circle
                bad = False
                for p in points:
                    if any(np.allclose(p, v) for v in poly.points):
                        continue  # skip triangle’s own vertices
                    if (p[0]-x_c)**2 + (p[1]-y_c)**2 < (r**2 - 1e-9):  # tolerance
                        bad = True
                        break

                if not bad:
                    # Deduplicate using a frozenset of tuples
                    key = frozenset(map(tuple, poly.points))
                    if key not in seen:
                        seen.add(key)
                        cleaned.append(poly)

            return cleaned
        
        self.polygons = clean_triangulation(polygon_list,self.cones)

    def order_cones(self, points):
        points = points.copy()
        
        # Find the point closest to car spawn position (0, 0)
        distances_to_origin = np.linalg.norm(points, axis=1)
        start_idx = np.argmin(distances_to_origin)
        
        ordered = [points[start_idx]]
        used = {start_idx}  # Use set instead of list
        for _ in range(len(points)-1):
            last = ordered[-1]
            # compute distances to unused points
            distance = np.linalg.norm(points - last, axis=1)

            #set distance of used poitns to infinity so they are not compared 
            distance[list(used)] = np.inf

            # least distance is new point
            nearest_point = np.argmin(distance)
            ordered.append(points[nearest_point])
            used.add(nearest_point)
        return np.array(ordered)

    def create_midpoints(self):
        midpoints = []

        #create polygon for all left edges and right edges seperately 
        left_edge = self.Polygon(self.order_cones(self.left_cones))
        right_edge = self.Polygon(self.order_cones(self.right_cones))

        # this assumes that we are always driving clockwise since the left edge barrier

        for tri in self.polygons:
            for edge in tri.edges:
                p1, p2 = edge
                
                midpoint = (p1 + p2) / 2

                #if inside the outer and not inside the inner barrier then it is within the track limits
                if left_edge.check_inside_polygon(midpoint) and not right_edge.check_inside_polygon(midpoint):
                    midpoints.append(midpoint)
        
        #filter the points that are too close to the left and right barriers
        clean = []
        tolerance = 1 # minimum distance tolerance to be inside the middle of the track

        for m in midpoints:
            d_left = np.min(np.linalg.norm(left_edge.points - m, axis=1))
            d_right = np.min(np.linalg.norm(right_edge.points - m, axis=1))
            if abs(d_left - d_right) < tolerance:
                clean.append(m)

        #there are duplicate values of each midpoint because of the del triangulation
        clean = np.unique(clean, axis=0)   
        # Order the midpoints using the same method as cones, starting from closest to (0,0)
        ordered_midpoints = self.order_cones(np.array(clean))
        
        # Reverse the order of midpoints to change direction
        ordered_midpoints = ordered_midpoints[::-1]

        return ordered_midpoints, left_edge, right_edge

    def spline_path(self, midpoints):
        
        #interpolate midpoint
        x = midpoints[:, 0]
        y = midpoints[:, 1]

        # create the spline object tck and the index u of path progress
        # per=1 makes it periodic (closed loop) - connects end back to start
        self.tck, self.u = splprep([x, y], s=0, per=1)

        # Calculate arc length using numerical integration
        u_samples = np.linspace(0, 1, 2000)
        dx, dy = splev(u_samples, self.tck, der=1)
        speed = np.sqrt(dx**2 + dy**2)
        self.arc_length = np.trapz(speed, u_samples)

    #umin and umax have to be in terms of the spline parameterization, 0 to 1
    #be careful to remove the scalar projection using the arc length

    def closest_point_on_spline(self,car_x, car_y ,tck, a , b):
        """Find the closest point on the spline to the given car position.
        
        Args:
            car_x (float): Car's x position
            car_y (float): Car's y position
            
        Returns:
            tuple: (u_value, distance_squared) where u_value is the parameter 
                   value on the spline and distance_squared is the squared distance
        """
            
        def dist_sq(u):
            xr, yr = splev(u, tck)
            return (xr - car_x)**2 + (yr - car_y)**2

        res = minimize_scalar(dist_sq, bounds=(a, b), method='bounded')
        return res.x, res.fun


    

        


    
    """rUSSELL COOOOOOODEEE       ENDDDDDDDDDDDDDDDDDDDD"""
        

    def R(self, theta):
        """SO(2) matrix constructor

        Args:
            theta (float): angle

        Returns:
            array: 2x2 rotation matrix for the given angle
        """
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    def _check_collision(self, state):
        return np.any(np.all(self.A@self.R(-state[2])@((self._cones - state[0:2]).T) < self.b[:, np.newaxis], axis=0), axis=0)
    def _get_accel(self, state, control):
        l = 0.79 # wheelbase=1.58, this is half
        return np.sqrt(
            control[0]**2
         + ((state[3]**2 / l) * np.sin(np.arctan(0.5 * np.tan(state[4]))))**2
        )
    def _check_accel(self, state, control):
        return self._get_accel(state, control) > 12 
    def run(self, tf=90):
        """Run the simulator with the given controller for `tf` seconds, storing results inside this Simulator.

        Args:
            tf (float, optional): how long to run the simulation. Defaults to 90.
        """
        self.log = []
        state = np.zeros(5)
        for t in np.arange(0, tf, 0.01):
            u = self.cb(state)
            assert isinstance(u, np.ndarray), f"expected numpy array from controller but got type {type(u)}"
            assert u.shape==(2,), f"expected shape (2,) from controller but received {u.shape}"
            u = np.array([
                np.clip(u[0], *self._accel_limits),
                np.clip(u[1], *self._steering_vel_limits)
            ])

            if ((state[4] > self._steering_limits[1] and u[1] > 0)
             or (state[4] < self._steering_limits[0] and u[1] < 0)):
                u[1] = 0
            crash = self._check_collision(state)
            slip = self._check_accel(state, u)
            self.log.append((t, state, u, crash, slip))
            state = self.dynamics(state, u).toarray().flatten()
    def get_results(self):
        """get the simulation results. gives a tuple of arrays: (timestamps, states, controls, crash, slip). 
        `crash` is a bool array which is true when the car is colliding with a cone.
        `slip` is a bool array which is true when the car is exceeding friction limits.

        Raises:
            ValueError: if the sim has not been run, there will be no results.

        Returns:
            (array, array, array, array, array): shapes (N,), (5, N), (2, N), (N,), (N,). time series data as described above.
        """
        try: log = self.log
        except: raise ValueError("cannot animate; no results exist. Did you .run() the simulator?")
        ts = np.array([i[0] for i in self.log])
        xs = np.concatenate([i[1][:, np.newaxis] for i in self.log], axis=1)
        us = np.concatenate([i[2][:, np.newaxis] for i in self.log], axis=1)
        crash  = np.array([i[3] for i in self.log])
        slip  = np.array([i[4] for i in self.log])
        return (ts, xs, us, crash, slip)
    def plot(self, block=True):
        """plot the last run of the simulator.

        Args:
            block (bool, optional): the `block` argument to plt.show(). Defaults to True.
        """
        ts, xs, us, crash, slip = self.get_results()
        fig, axs = plt.subplots(7, sharex=True)

        axs[0].plot(ts, xs[0]); axs[0].set_ylabel('x pos (m)')
        axs[1].plot(ts, xs[1]); axs[1].set_ylabel('y pos (m)')
        axs[2].plot(ts, xs[2]); axs[2].set_ylabel('heading (rad)')
        axs[3].plot(ts, xs[3]); axs[3].set_ylabel('velocity (m/s)')
        axs[4].plot(ts, xs[4]); axs[4].set_ylabel('steering angle (rad)')

        axs[5].plot(ts, us[0]); axs[5].set_ylabel('fwd accel (m/s^2)')
        axs[6].plot(ts, us[1]); axs[6].set_ylabel('steering velocity (rad/s)')
        for i in range(5):
            axs[i].scatter(ts[crash], xs[i, crash], color='tab:red', marker='+')
            axs[i].scatter(ts[slip], xs[i, slip], color='tab:orange', marker='x')
        for i in range(2):
            axs[5+i].scatter(ts[crash], us[i, crash], color='tab:red', marker='+')
            axs[5+i].scatter(ts[slip], us[i, slip], color='tab:orange', marker='x')
        plt.show(block=block)
    def animate(self, save=False, filename='sim.gif', block=True):
        """animate the most recent run of the simulator.

        Args:
            save (bool, optional): whether or not to save the animation to a gif. Defaults to False.
            filename (str, optional): filename to save as, if `save` is True. Defaults to 'sim.gif'.
            block (bool, optional): the `block` argument to plt.show(). Defaults to True.
        """
        # Create the spline path if it doesn't exist
        if not hasattr(self, 'tck'):
            self.del_triangulation()
            midpoints, _, _ = self.create_midpoints()
            self.spline_path(midpoints)
        
        fig = plt.figure(figsize=(10, 12))
        axs = fig.subplots(2, 1, height_ratios=[4, 1])
        axs[0].set_aspect('equal')
        axs[1].set_aspect(1)
        axs[0].scatter(*self.left_cones.T, color='tab:blue')
        axs[0].scatter(*self.right_cones.T, color='tab:orange')
        
        # Plot the spline path
        s_values = np.linspace(0, 1, 200)
        spline_points = np.array([splev(s, self.tck) for s in s_values])
        axs[0].plot(spline_points[:, 0], spline_points[:, 1], 'g-', 
                   linewidth=2, alpha=0.7, label='Racing Line')
        
        outline = axs[0].add_patch(patches.Polygon(self.car_outline, fill=True, closed=True, facecolor='lightblue', edgecolor='black'))
        posearrow = axs[0].add_patch(patches.FancyArrow(0, 0, 1, 0, width=0.1, color='tab:red'))
        
        # Add closest point marker
        closest_point_marker = axs[0].scatter([], [], color='red', s=100, marker='o', zorder=10, label='Closest Point')
        
        # Add car position text
        car_position_text = axs[0].text(0.02, 0.98, '', transform=axs[0].transAxes, 
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        accel = axs[1].plot([0],[0])[0]
        axs[0].set_title('car pose')
        axs[0].legend()

        axs[1].set_title('net acceleration')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('acceleration (m/s^2)')

        ts, xs, us, crash, slip = self.get_results()
        accel_values = np.array([self._get_accel(x, u) for x, u in zip(xs.T, us.T)])
        axs[1].hlines([12], [0], [np.max(ts)], linestyles='dashed', color='tab:red')
        collision_patches = []
        
        def frame(i):
            outline_points = ((self.R(xs[2, i])@self.car_outline.T).T + xs[0:2, i])
            arrow_data = dict(
                x=xs[0, i],
                y=xs[1, i],
                dx=np.cos(xs[2, i]),
                dy=np.sin(xs[2, i]),
            )
            
            # Find and update closest point on spline
            car_x, car_y = xs[0, i], xs[1, i]
            
            # Update car position text
            car_position_text.set_text(f'Car Position:\nX: {car_x:.2f} m\nY: {car_y:.2f} m\nHeading: {xs[2, i]:.2f} rad\nVelocity: {xs[3, i]:.2f} m/s\nSteering: {xs[4, i]:.2f} rad')
            
            # Calculate closest point for visualization
            u_closest, _ = self.closest_point_on_spline(car_x, car_y, self.tck, 0.0, 1.0)
            if u_closest is not None:
                closest_x, closest_y = splev(u_closest, self.tck)
                closest_point_marker.set_offsets([[closest_x, closest_y]])
            
            if self._check_collision(xs[:, i]):
                collision_car = patches.Polygon(outline_points, color='tab:red', fill=False, linewidth=0.1)
                collision_pose = patches.FancyArrow(**arrow_data, width=0.1, color='black')
                collision_patches.append(axs[0].add_patch(collision_car))
                collision_patches.append(axs[0].add_patch(collision_pose))
            outline.set_xy(outline_points)
            posearrow.set_data(**arrow_data)
            accel.set_xdata(ts[:i+1])
            accel.set_ydata(accel_values[:i+1])
            return [accel, outline, posearrow, closest_point_marker, car_position_text] + collision_patches

        anim = animation.FuncAnimation(fig, frame, len(ts), interval=10)
        if save:
            anim.save(filename, writer='ffmpeg')
        plt.show(block=block)


if __name__ == '__main__':
    sim = Simulator(lambda x: np.array([1, -0.1*(x[4]-(-0.25))]))
    print(sim.centerline(1.0))
    print(sim.centerline(np.array([1.0, 2.0])))
    print(sim.centerline([1.0, 2.0, 3, 4]))
    sim.run()
    sim.animate()
    sim.plot()