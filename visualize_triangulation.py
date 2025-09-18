import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from simulator import Simulator
from scipy.spatial import Delaunay
from scipy.interpolate import splev
import matplotlib.animation as animation
import time

def plot_polygon(ax, vertices, color='blue', alpha=0.3, label=None, linewidth=1):
    """Helper function to plot a polygon on the given axis."""
    if len(vertices) < 3:
        print(f"Warning: Need at least 3 vertices for polygon, got {len(vertices)}")
        return
    
    # Create matplotlib polygon
    polygon = MplPolygon(vertices, closed=True, facecolor=color, 
                        edgecolor='black', alpha=alpha, linewidth=linewidth)
    ax.add_patch(polygon)
    
    # Add to legend if label provided
    if label:
        # Create a proxy artist for the legend
        ax.plot([], [], color=color, alpha=alpha, linewidth=3, label=label)
    
    # Plot vertices as small dots
    ax.scatter(vertices[:, 0], vertices[:, 1], color='black', s=20, alpha=0.8, zorder=5)

def create_track_boundary(cones):
    """Create a boundary polygon from cone positions."""
    # Simple convex hull approximation
    center = np.mean(cones, axis=0)
    
    # Sort cones by angle from center to create a boundary
    angles = np.arctan2(cones[:, 1] - center[1], cones[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    # Take every few points to avoid too complex polygon
    step = max(1, len(sorted_indices) // 12)  # Max 12 boundary points
    boundary_indices = sorted_indices[::step]
    
    return cones[boundary_indices]

def visualize_midpoint_order():
    """Debug visualization to show the order of midpoints"""
    sim = Simulator()
    
    # Create triangulation and get midpoints
    sim.del_triangulation()
    midpoints, left_edge, right_edge = sim.create_midpoints()
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_aspect('equal')
    
    # Plot track boundaries
    ax.scatter(*sim.left_cones.T, color='tab:blue', label='Left Cones', alpha=0.6)
    ax.scatter(*sim.right_cones.T, color='tab:orange', label='Right Cones', alpha=0.6)
    
    # Plot midpoints with numbered labels
    ax.scatter(midpoints[:, 0], midpoints[:, 1], color='purple', s=50, label='Midpoints', zorder=5)
    
    # Add number labels to each midpoint
    for i, (x, y) in enumerate(midpoints):
        ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, color='black', weight='bold',
                   bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw lines connecting consecutive midpoints to show order
    for i in range(len(midpoints) - 1):
        ax.plot([midpoints[i, 0], midpoints[i+1, 0]], 
                [midpoints[i, 1], midpoints[i+1, 1]], 
                'r-', alpha=0.5, linewidth=1.5)
    
    # Connect last point to first to show if it's a closed loop
    ax.plot([midpoints[-1, 0], midpoints[0, 0]], 
            [midpoints[-1, 1], midpoints[0, 1]], 
            'r--', alpha=0.5, linewidth=2, label='Loop closure')
    
    # Create and visualize the spline from simulator
    sim.spline_path(midpoints)
    
    # Generate spline points for visualization
    s_values = np.linspace(0, 1, 200)
    spline_points = np.array([splev(s, sim.tck) for s in s_values])
    
    # Plot the spline
    ax.plot(spline_points[:, 0], spline_points[:, 1], 'g-', 
            linewidth=3, alpha=0.8, label='Spline Path', zorder=10)
    
    ax.set_title(f'Midpoint Order Visualization ({len(midpoints)} points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    
    # Set axis limits to match main visualization
    ax.set_xlim(-20, 35)
    ax.set_ylim(-35, 10)
    
    plt.tight_layout()
    plt.show()

def visualize_all_triangulated_polygons():
    """Visualize all triangulated polygons from the simulator."""
    sim = Simulator()
    
    # Run triangulation
    print("Running Delaunay triangulation...")
    sim.del_triangulation()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot all cones first
    ax.scatter(sim.cones[:, 0], sim.cones[:, 1], 
              color='red', s=60, alpha=0.8, label='Cones', zorder=10)
    
    # Plot all triangulated polygons
    if hasattr(sim, 'polygons') and sim.polygons:
        print(f"Plotting {len(sim.polygons)} triangulated polygons...")
        
        # Use the same color for all triangles
        triangle_color = 'lightblue'
        
        # for i, simplex in enumerate(Delaunay(sim.cones).simplices):
        #     plot_polygon(ax, sim.cones[simplex], color=triangle_color, alpha=0.1, 
        #                label='Triangles' if i == 0 else None, linewidth=1.5)
            
        for i, polygon in enumerate(sim.polygons):
            plot_polygon(ax, polygon.points, color=triangle_color, alpha=0.1, 
                       label='Triangles' if i == 0 else None, linewidth=1.5)
            
        # Plot midpoints
        midpoints, left_edge, right_edge = sim.create_midpoints()
        if midpoints.size > 0:
            ax.scatter(midpoints[:, 0], midpoints[:, 1], color='purple', s=40, alpha=0.8, label='Midpoints', zorder=15)
            
            # Plot splined path
            
            # spline_points = sim.spline_path(midpoints)
            
            # ax.plot(spline_points[:, 0], spline_points[:, 1], color='green', linewidth=3, alpha=0.8, label='Splined Path', zorder=12)
        
         
    else:
        print("No triangulated polygons found!")
        return
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'All Triangulated Polygons ({len(sim.polygons)} triangles)')
    
    # Set axis limits
    ax.set_xlim(-20, 35)
    ax.set_ylim(-35, 10)
    
    # Add statistics
    stats_text = f"""
    Triangulation Results:
    • Total triangles: {len(sim.polygons)}
    • Total cones: {len(sim.cones)}
    • Triangles per cone: {len(sim.polygons)/len(sim.cones):.2f}
    """
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    visualize_midpoint_order()
    visualize_all_triangulated_polygons()
   

