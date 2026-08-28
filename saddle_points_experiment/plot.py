import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results.json')
    parser.add_argument('--out_dir', type=str, default='plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    dimensions = sorted([int(k) for k in data.keys()])
    
    # -------------------------------------------------------------
    # Plot 1: Distribution of alpha (fraction of negative eigenvalues) vs Dimension
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Custom styling for premium feel
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gather data for plotting
    plot_data = []
    labels = []
    
    for d in dimensions:
        pts = data[str(d)]
        alphas = [pt['alpha'] for pt in pts]
        plot_data.append(alphas)
        labels.append(str(d))
        
        # Add jittered points for scatter overlay
        x_jitter = np.random.normal(dimensions.index(d) + 1, 0.04, size=len(alphas))
        ax.scatter(x_jitter, alphas, alpha=0.4, color='#3b82f6', edgecolor='none', zorder=2)
        
    # Create boxplot supporting both old and new Matplotlib versions
    try:
        bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, zorder=3,
                        showfliers=False, widths=0.5)
    except TypeError:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, zorder=3,
                        showfliers=False, widths=0.5)
    
    # Style boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('#ffffff99')
        patch.set_edgecolor('#1e3a8a')
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#ef4444')
        median.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_color('#1e3a8a')
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color('#1e3a8a')
        cap.set_linewidth(1)
        
    ax.set_title('Distribution of Fraction of Negative Eigenvalues (Index) vs. Dimension', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dimension (D)', fontsize=12, labelpad=10)
    ax.set_ylabel('Fraction of Negative Eigenvalues (α)', fontsize=12, labelpad=10)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'alpha_distribution.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Probability of Local Minima (alpha == 0) vs Dimension
    # -------------------------------------------------------------
    minima_probs = []
    for d in dimensions:
        pts = data[str(d)]
        if len(pts) == 0:
            minima_probs.append(0.0)
            continue
        # Count critical points that are local minima (alpha == 0)
        num_minima = sum(1 for pt in pts if pt['alpha'] == 0.0)
        minima_probs.append(num_minima / len(pts))
        
    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot experimental results
    ax.plot(dimensions, minima_probs, 'o-', color='#3b82f6', linewidth=2.5, markersize=8, label='Empirical probability')
    
    # Plot theoretical curve (1/2)^D for reference
    d_fine = np.linspace(min(dimensions), max(dimensions), 200)
    theoretical = (0.5) ** d_fine
    ax.plot(d_fine, theoretical, '--', color='#ef4444', linewidth=1.5, label='Theoretical independent $(0.5)^D$')
    
    ax.set_xscale('log')
    ax.set_xticks(dimensions)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_title('Probability of a Critical Point Being a Local Minimum', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dimension D (log scale)', fontsize=12, labelpad=10)
    ax.set_ylabel('Probability P(α = 0)', fontsize=12, labelpad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'minima_probability.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Function Value / Loss vs. Index (alpha)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Use a subset of dimensions for readability
    selected_dims = [2, 5, 10, 20, 50]
    colors = ['#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899']
    
    for idx, d in enumerate(selected_dims):
        if str(d) not in data:
            continue
        pts = data[str(d)]
        alphas = [pt['alpha'] for pt in pts]
        vals = [pt['val'] for pt in pts]
        
        ax.scatter(alphas, vals, alpha=0.7, color=colors[idx % len(colors)], label=f'D = {d}', edgecolors='none', s=40)
        
    ax.set_title('Function Value vs. Fraction of Negative Eigenvalues', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Fraction of Negative Eigenvalues (α)', fontsize=12, labelpad=10)
    ax.set_ylabel('Function Value f(x*)', fontsize=12, labelpad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', title='Dimension')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'value_vs_alpha.png'), dpi=300)
    plt.close()
    
    print(f"Plots successfully generated and saved to {args.out_dir}/")

if __name__ == '__main__':
    main()
