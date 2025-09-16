import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from simulator import Simulator
from scipy.spatial import Delaunay
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
            
         
    else:
        print("No triangulated polygons found!")
        return
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'All Triangulated Polygons ({len(sim.polygons)} triangles)')
    
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
  
    visualize_all_triangulated_polygons()
    
    
  