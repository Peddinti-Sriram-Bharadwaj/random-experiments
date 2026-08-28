import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 - 0.5 * y**2 + 0.25 * y**4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='optimization_results.json')
    parser.add_argument('--out_dir', type=str, default='plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(args.input, 'r') as f_in:
        results = json.load(f_in)
        
    x0 = results['x0']
    
    # Extract trajectory history for each optimizer
    methods = ['GD', 'GDM', 'Newton', 'SFN']
    colors = {
        'GD': '#f59e0b',      # Orange
        'GDM': '#3b82f6',     # Blue
        'Newton': '#ef4444',  # Red
        'SFN': '#10b981'      # Green
    }
    markers = {
        'GD': 'o',
        'GDM': '^',
        'Newton': 'x',
        'SFN': 'd'
    }
    
    data = {}
    for m in methods:
        hist = results[m]
        data[m] = {
            'x': np.array([pt['x'] for pt in hist]),
            'y': np.array([pt['y'] for pt in hist]),
            'val': np.array([pt['val'] for pt in hist]),
            'grad_norm': np.array([pt['grad_norm'] for pt in hist])
        }
        
    # Custom styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # Plot 1: 2D Contour Plot with Trajectories
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Grid for contour
    x_grid = np.linspace(-0.25, 0.25, 200)
    y_grid = np.linspace(-0.15, 1.15, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)
    
    # Draw contour background
    contour = ax.contourf(X, Y, Z, levels=50, cmap='Spectral_r', alpha=0.85)
    fig.colorbar(contour, ax=ax, label='Loss / Function Value $f(x,y)$')
    
    # Plot trajectories
    for m in methods:
        ax.plot(data[m]['x'], data[m]['y'], color=colors[m], label=m, linewidth=2.0, zorder=4)
        ax.scatter(data[m]['x'][::5], data[m]['y'][::5], color=colors[m], marker=markers[m], s=35, zorder=5) # Mark every 5 steps
        
    # Mark special points
    ax.scatter(x0[0], x0[1], color='black', marker='*', s=150, zorder=6, label='Start $x_0$')
    ax.scatter(0, 0, color='white', edgecolor='black', marker='o', s=80, zorder=6, label='Saddle Point (0,0)')
    ax.scatter(0, 1.0, color='gold', edgecolor='black', marker='X', s=100, zorder=6, label='Local Min (0,1)')
    
    ax.set_title('Optimization Trajectories Near Saddle Point (2D)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('$x$ (Positive curvature direction)', fontsize=12)
    ax.set_ylabel('$y$ (Negative curvature direction)', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.15, 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'contour_trajectory.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: 3D Surface Plot with Trajectories
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D Grid
    x_grid_3d = np.linspace(-0.25, 0.25, 100)
    y_grid_3d = np.linspace(-0.15, 1.15, 100)
    X3d, Y3d = np.meshgrid(x_grid_3d, y_grid_3d)
    Z3d = f(X3d, Y3d)
    
    # Plot surface
    surf = ax.plot_surface(X3d, Y3d, Z3d, cmap='Spectral_r', alpha=0.5, edgecolor='none', rstride=2, cstride=2)
    
    # Plot 3D trajectories
    for m in methods:
        ax.plot(data[m]['x'], data[m]['y'], data[m]['val'], color=colors[m], label=m, linewidth=3.0, zorder=10)
        ax.scatter(data[m]['x'][::10], data[m]['y'][::10], data[m]['val'][::10], color=colors[m], marker=markers[m], s=30, zorder=11)
        
    # Mark start and saddle points
    ax.scatter(x0[0], x0[1], f(x0[0], x0[1]), color='black', marker='*', s=150, zorder=12, label='Start $x_0$')
    ax.scatter(0, 0, f(0, 0), color='white', edgecolor='black', marker='o', s=80, zorder=12, label='Saddle Point')
    ax.scatter(0, 1.0, f(0, 1.0), color='gold', edgecolor='black', marker='X', s=100, zorder=12, label='Local Min')
    
    ax.set_title('Optimization Trajectories Over Landscape (3D)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_zlabel('$f(x,y)$', fontsize=12)
    
    # Adjust camera angle for best visual clarity
    ax.view_init(elev=35, azim=-45)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'surface_trajectory_3d.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Loss vs. Iterations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for m in methods:
        ax.plot(data[m]['val'], color=colors[m], label=m, linewidth=2.0)
        
    ax.set_title('Function Value $f(x,y)$ over Iterations', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Iteration Step', fontsize=12)
    ax.set_ylabel('Function Value (Loss)', fontsize=12)
    ax.set_ylim(-0.3, 0.05)
    ax.legend(frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'loss_vs_iteration.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 4: Gradient Norm vs. Iterations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for m in methods:
        ax.plot(data[m]['grad_norm'], color=colors[m], label=m, linewidth=2.0)
        
    ax.set_title('Gradient Norm over Iterations', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Iteration Step', fontsize=12)
    ax.set_ylabel('$\\|\\nabla f(x,y)\\|_2$', fontsize=12)
    ax.set_yscale('log')
    ax.legend(frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'gradient_norm_vs_iteration.png'), dpi=300)
    plt.close()
    
    print(f"Escaping saddle points plots generated in {args.out_dir}/")

if __name__ == '__main__':
    main()
