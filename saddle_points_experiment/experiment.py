import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import time

class RandomLandscape(nn.Module):
    def __init__(self, d, m=100, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        # Random parameters (fixed)
        # Weights w: (m, d)
        self.w = nn.Parameter(torch.randn(m, d) / np.sqrt(d), requires_grad=False)
        self.b = nn.Parameter(torch.randn(m), requires_grad=False)
        self.v = nn.Parameter(torch.randn(m), requires_grad=False)
        
    def forward(self, x):
        # x is (d,)
        lin = torch.matmul(self.w, x) + self.b
        return torch.sum(self.v * torch.tanh(lin))

def loss_fn(x, landscape):
    val = landscape(x)
    grad = torch.autograd.grad(val, x, create_graph=True)[0]
    loss = 0.5 * torch.sum(grad ** 2)
    return loss, grad, val

def get_hessian(landscape, x):
    # Compute the Hessian matrix
    H = torch.autograd.functional.hessian(landscape, x)
    return H

def find_critical_points(d, num_starts=300, seed=42):
    print(f"--- Running dimension D = {d} with {num_starts} starts ---")
    landscape = RandomLandscape(d, m=100, seed=seed)
    critical_points = []
    
    # Set a stable seed for starting points
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    
    start_time = time.time()
    
    for start_idx in range(num_starts):
        # Sample start point with varying scale to explore different regions
        scale = np.random.uniform(0.5, 3.5)
        x = torch.randn(d) * scale
        x.requires_grad_(True)
        
        # Optimize the gradient norm squared
        optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
        
        converged = False
        
        # Run optimizer
        def closure():
            optimizer.zero_grad()
            loss, _, _ = loss_fn(x, landscape)
            loss.backward()
            return loss
            
        try:
            optimizer.step(closure)
            
            # Check convergence without blocking autograd
            x_eval = x.clone().detach().requires_grad_(True)
            val = landscape(x_eval)
            grad = torch.autograd.grad(val, x_eval)[0]
            grad_norm = torch.norm(grad).item()
            
            # Use 1.5e-3 threshold to robustly capture critical points in higher dimensions
            if grad_norm < 1.5e-3:
                converged = True
        except Exception as e:
            # Handle numerical instability if any (e.g. line search failures)
            continue
            
        if converged:
            critical_points.append({
                'x': x_eval.clone().detach(),
                'val': val.item(),
                'grad_norm': grad_norm
            })
            
    # Deduplicate the found critical points
    unique_points = []
    for pt in critical_points:
        is_dup = False
        for upt in unique_points:
            # If coordinates are very close, they represent the same critical point
            if torch.norm(pt['x'] - upt['x']) < 1e-2:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(pt)
            
    print(f"Found {len(critical_points)} candidate points. Unique critical points: {len(unique_points)}")
    
    # Compute Hessians and eigenvalues
    results = []
    for pt in unique_points:
        x_val = pt['x']
        H = get_hessian(landscape, x_val)
        eigenvals, _ = torch.linalg.eigh(H)
        eigenvals_np = eigenvals.numpy().tolist()
        
        # Count negative eigenvalues.
        # We use a tiny negative threshold to avoid counting numerical zeros as negative.
        num_neg = sum(1 for ev in eigenvals_np if ev < -1e-6)
        alpha = num_neg / d
        
        results.append({
            'val': pt['val'],
            'grad_norm': pt['grad_norm'],
            'alpha': alpha,
            'eigenvalues': eigenvals_np,
            'x': x_val.numpy().tolist()
        })
        
    elapsed = time.time() - start_time
    print(f"Dimension {d} completed in {elapsed:.2f}s. Unique critical points: {len(results)}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, nargs='+', default=[2, 3, 5, 10, 20, 50, 100])
    parser.add_argument('--starts', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results.json')
    args = parser.parse_args()
    
    all_results = {}
    for d in args.dimensions:
        # For higher dimensions, we can reduce the number of starts to keep it fast
        starts = args.starts
        if d >= 50:
            starts = min(starts, 150)
        if d >= 100:
            starts = min(starts, 100)
            
        res = find_critical_points(d, num_starts=starts, seed=args.seed)
        all_results[str(d)] = res
        
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()
