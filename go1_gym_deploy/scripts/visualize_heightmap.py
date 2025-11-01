"""
Visualize the terrain heightmap from depth estimation.

This script creates a visual representation of the heightmap that the policy sees.
Useful for debugging and understanding what terrain information the robot perceives.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle
import argparse


def create_heightmap_visualization(heightmap, grid_x, grid_y, title="Terrain Heightmap"):
    """
    Create a visualization of the terrain heightmap.
    
    Args:
        heightmap: (N,) flattened heightmap
        grid_x: X coordinates of measurement points
        grid_y: Y coordinates of measurement points
        title: Plot title
    """
    # Reshape heightmap
    heightmap_2d = heightmap.reshape(len(grid_x), len(grid_y))
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Main heightmap plot
    ax1 = plt.subplot(2, 2, 1)
    im = ax1.imshow(heightmap_2d, cmap='terrain', origin='lower', 
                    extent=[grid_y[0], grid_y[-1], grid_x[0], grid_x[-1]],
                    aspect='auto')
    ax1.set_xlabel('Lateral (m)')
    ax1.set_ylabel('Forward (m)')
    ax1.set_title(f'{title}\n(Top-down view)')
    ax1.grid(True, alpha=0.3)
    
    # Add robot position
    robot_marker = Circle((0, 0), 0.05, color='red', zorder=10, label='Robot')
    ax1.add_patch(robot_marker)
    ax1.legend()
    
    plt.colorbar(im, ax=ax1, label='Height (m)')
    
    # 3D surface plot
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    X, Y = np.meshgrid(grid_y, grid_x)
    surf = ax2.plot_surface(X, Y, heightmap_2d, cmap='terrain', 
                           linewidth=0, antialiased=True, alpha=0.8)
    ax2.set_xlabel('Lateral (m)')
    ax2.set_ylabel('Forward (m)')
    ax2.set_zlabel('Height (m)')
    ax2.set_title('3D Terrain Surface')
    
    # Height distribution
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(heightmap.flatten(), bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Height (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Height Distribution')
    ax3.axvline(heightmap.mean(), color='red', linestyle='--', label=f'Mean: {heightmap.mean():.3f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
    Heightmap Statistics
    {'='*40}
    
    Grid Size: {len(grid_x)} x {len(grid_y)} = {len(heightmap)} points
    
    X Range: [{grid_x[0]:.2f}, {grid_x[-1]:.2f}] m
    Y Range: [{grid_y[0]:.2f}, {grid_y[-1]:.2f}] m
    
    Height Statistics:
      Min:    {heightmap.min():.3f} m
      Max:    {heightmap.max():.3f} m
      Mean:   {heightmap.mean():.3f} m
      Std:    {heightmap.std():.3f} m
      Median: {np.median(heightmap):.3f} m
    
    Terrain Roughness:
      Std Dev:  {heightmap.std():.3f} m
      Range:    {heightmap.max() - heightmap.min():.3f} m
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def create_example_heightmaps():
    """Create example heightmaps for demonstration"""
    
    # Grid configuration (same as default in code)
    grid_x = np.linspace(-0.5, 0.5, 17)
    grid_y = np.linspace(-0.5, 0.5, 11)
    
    X, Y = np.meshgrid(grid_y, grid_x)
    
    examples = []
    
    # 1. Flat terrain
    flat = np.zeros((len(grid_x), len(grid_y)))
    examples.append(("Flat Terrain", flat.flatten()))
    
    # 2. Step
    step = np.zeros((len(grid_x), len(grid_y)))
    step[grid_x > 0.2] = 0.1
    examples.append(("Step Terrain (10cm)", step.flatten()))
    
    # 3. Slope
    slope = X * 0.2  # 20cm slope over 1m
    examples.append(("Sloped Terrain", slope.flatten()))
    
    # 4. Rough terrain
    rough = np.random.randn(len(grid_x), len(grid_y)) * 0.05
    examples.append(("Rough Terrain (random)", rough.flatten()))
    
    # 5. Wave
    wave = 0.05 * np.sin(X * 10) * np.cos(Y * 10)
    examples.append(("Wave Pattern", wave.flatten()))
    
    return examples, grid_x, grid_y


def load_heightmap_from_file(filepath):
    """Load heightmap from numpy file"""
    data = np.load(filepath)
    
    # Try different possible formats
    if 'heightmap' in data:
        heightmap = data['heightmap']
    elif 'measured_heights' in data:
        heightmap = data['measured_heights']
    else:
        # Assume it's just the array
        heightmap = data
    
    if len(heightmap.shape) == 2:
        heightmap = heightmap.flatten()
    
    return heightmap


def main():
    parser = argparse.ArgumentParser(description='Visualize terrain heightmap')
    
    parser.add_argument('--heightmap_file', type=str,
                        help='Path to heightmap numpy file (.npy or .npz)')
    parser.add_argument('--examples', action='store_true',
                        help='Show example heightmaps')
    parser.add_argument('--grid_x_min', type=float, default=-0.5,
                        help='Grid X minimum')
    parser.add_argument('--grid_x_max', type=float, default=0.5,
                        help='Grid X maximum')
    parser.add_argument('--grid_x_num', type=int, default=17,
                        help='Number of X grid points')
    parser.add_argument('--grid_y_min', type=float, default=-0.5,
                        help='Grid Y minimum')
    parser.add_argument('--grid_y_max', type=float, default=0.5,
                        help='Grid Y maximum')
    parser.add_argument('--grid_y_num', type=int, default=11,
                        help='Number of Y grid points')
    parser.add_argument('--save', type=str,
                        help='Save figure to file')
    
    args = parser.parse_args()
    
    # Create grid
    grid_x = np.linspace(args.grid_x_min, args.grid_x_max, args.grid_x_num)
    grid_y = np.linspace(args.grid_y_min, args.grid_y_max, args.grid_y_num)
    
    if args.examples:
        # Show example heightmaps
        examples, grid_x, grid_y = create_example_heightmaps()
        
        for title, heightmap in examples:
            fig = create_heightmap_visualization(heightmap, grid_x, grid_y, title)
            if args.save:
                save_path = args.save.replace('.png', f'_{title.replace(" ", "_")}.png')
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    elif args.heightmap_file:
        # Load and visualize heightmap from file
        print(f"Loading heightmap from: {args.heightmap_file}")
        heightmap = load_heightmap_from_file(args.heightmap_file)
        
        print(f"✓ Loaded heightmap: shape={heightmap.shape}")
        
        # Check if size matches grid
        expected_size = len(grid_x) * len(grid_y)
        if len(heightmap) != expected_size:
            print(f"⚠️  Warning: Heightmap size {len(heightmap)} doesn't match grid {expected_size}")
            print(f"   Adjusting grid to match...")
            # Try to infer grid size
            n = int(np.sqrt(len(heightmap)))
            if n * n == len(heightmap):
                grid_x = np.linspace(args.grid_x_min, args.grid_x_max, n)
                grid_y = np.linspace(args.grid_y_min, args.grid_y_max, n)
                print(f"   Using grid: {n} x {n}")
            else:
                print(f"   Cannot infer grid, using default")
        
        fig = create_heightmap_visualization(heightmap, grid_x, grid_y, 
                                            f"Heightmap: {args.heightmap_file}")
        
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {args.save}")
        
        plt.show()
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Show example heightmaps")
        print("  python visualize_heightmap.py --examples")
        print("")
        print("  # Visualize heightmap from file")
        print("  python visualize_heightmap.py --heightmap_file data.npy")
        print("")
        print("  # Custom grid and save")
        print("  python visualize_heightmap.py --heightmap_file data.npy \\")
        print("                                 --grid_x_num 21 --grid_y_num 15 \\")
        print("                                 --save output.png")


if __name__ == "__main__":
    main()
