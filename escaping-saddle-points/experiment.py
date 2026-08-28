import numpy as np
import json
import os
import argparse

def f(x, y):
    # Loss landscape with a saddle point at (0, 0) and local minima at (0, -1) and (0, 1)
    return x**2 - 0.5 * y**2 + 0.25 * y**4

def grad_f(x, y):
    # Gradient vector
    return np.array([2.0 * x, -y + y**3])

def hessian_f(x, y):
    # Hessian matrix (diagonal for this landscape)
    return np.array([
        [2.0, 0.0],
        [0.0, -1.0 + 3.0 * y**2]
    ])

def optimize_gd(x0, lr=0.1, steps=200):
    history = []
    xy = np.array(x0, dtype=float)
    for _ in range(steps):
        val = f(xy[0], xy[1])
        grad = grad_f(xy[0], xy[1])
        history.append({
            'x': xy[0],
            'y': xy[1],
            'val': val,
            'grad_norm': np.linalg.norm(grad)
        })
        xy -= lr * grad
    return history

def optimize_gdm(x0, lr=0.1, beta=0.9, steps=200):
    history = []
    xy = np.array(x0, dtype=float)
    v = np.zeros(2)
    for _ in range(steps):
        val = f(xy[0], xy[1])
        grad = grad_f(xy[0], xy[1])
        history.append({
            'x': xy[0],
            'y': xy[1],
            'val': val,
            'grad_norm': np.linalg.norm(grad)
        })
        v = beta * v + lr * grad
        xy -= v
    return history

def optimize_newton(x0, lr=1.0, steps=200, epsilon=1e-5):
    history = []
    xy = np.array(x0, dtype=float)
    for _ in range(steps):
        val = f(xy[0], xy[1])
        grad = grad_f(xy[0], xy[1])
        history.append({
            'x': xy[0],
            'y': xy[1],
            'val': val,
            'grad_norm': np.linalg.norm(grad)
        })
        H = hessian_f(xy[0], xy[1])
        
        # Invert Hessian with damping for numerical stability
        inv_H = np.zeros((2, 2))
        inv_H[0, 0] = 1.0 / H[0, 0]
        # Avoid dividing by zero if Hessian element is very close to 0
        h22 = H[1, 1]
        if abs(h22) < epsilon:
            h22_reg = h22 + np.sign(h22) * epsilon if h22 != 0 else epsilon
        else:
            h22_reg = h22
        inv_H[1, 1] = 1.0 / h22_reg
        
        xy -= lr * (inv_H @ grad)
    return history

def optimize_sfn(x0, lr=1.0, steps=200, epsilon=1e-5):
    # Saddle-Free Newton (Dauphin et al. 2014)
    history = []
    xy = np.array(x0, dtype=float)
    for _ in range(steps):
        val = f(xy[0], xy[1])
        grad = grad_f(xy[0], xy[1])
        history.append({
            'x': xy[0],
            'y': xy[1],
            'val': val,
            'grad_norm': np.linalg.norm(grad)
        })
        H = hessian_f(xy[0], xy[1])
        
        # Take the absolute value of the Hessian eigenvalues
        abs_H = np.zeros((2, 2))
        abs_H[0, 0] = abs(H[0, 0])
        abs_H[1, 1] = abs(H[1, 1])
        
        # Invert the absolute Hessian with damping
        inv_abs_H = np.zeros((2, 2))
        inv_abs_H[0, 0] = 1.0 / abs_H[0, 0]
        
        h22_abs = abs_H[1, 1]
        if h22_abs < epsilon:
            h22_abs = epsilon
        inv_abs_H[1, 1] = 1.0 / h22_abs
        
        xy -= lr * (inv_abs_H @ grad)
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x0', type=float, nargs=2, default=[0.1, 0.001])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr_gd', type=float, default=0.1)
    parser.add_argument('--lr_gdm', type=float, default=0.1)
    parser.add_argument('--lr_newton', type=float, default=1.0)
    parser.add_argument('--lr_sfn', type=float, default=1.0)
    parser.add_argument('--out', type=str, default='optimization_results.json')
    args = parser.parse_args()
    
    print(f"Starting experiment with x0 = {args.x0}")
    
    gd_history = optimize_gd(args.x0, lr=args.lr_gd, steps=args.steps)
    gdm_history = optimize_gdm(args.x0, lr=args.lr_gdm, steps=args.steps)
    newton_history = optimize_newton(args.x0, lr=args.lr_newton, steps=args.steps)
    sfn_history = optimize_sfn(args.x0, lr=args.lr_sfn, steps=args.steps)
    
    results = {
        'x0': args.x0,
        'GD': gd_history,
        'GDM': gdm_history,
        'Newton': newton_history,
        'SFN': sfn_history
    }
    
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results successfully saved to {args.out}")

if __name__ == '__main__':
    main()
