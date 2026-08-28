# Escaping Saddle Points in Optimization Landscapes

This experiment demonstrates how optimization algorithms behave in the vicinity of a strict saddle point, illustrating the "plateau problem" discussed in Chapter 8 of the *Deep Learning* textbook and the paper by **Dauphin et al. (2014)** (*Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*).

We compare four optimization routines:
1. **Gradient Descent (GD)** (stalls on plateaus)
2. **Gradient Descent with Momentum (GDM)** (accelerates escape)
3. **Standard Newton's Method** (gets attracted to and trapped at saddle points)
4. **Saddle-Free Newton (SFN)** (proposed by Dauphin et al., escapes saddle points instantly)

---

## The Optimization Landscape

We use a 2D non-convex landscape with a strict saddle point at $(0,0)$ and two local minima at $(0, \pm 1)$:
$$f(x, y) = x^2 - \frac{1}{2}y^2 + \frac{1}{4}y^4$$

### Mathematical Formulations

- **Gradient**:
  $$\nabla f(x, y) = \begin{bmatrix} 2x \\ -y + y^3 \end{bmatrix}$$
- **Hessian**:
  $$H(x, y) = \begin{bmatrix} 2 & 0 \\ 0 & -1 + 3y^2 \end{bmatrix}$$
  At $(0,0)$, $H$ has eigenvalues $\lambda_1 = +2$ (positive curvature along $x$) and $\lambda_2 = -1$ (negative curvature along $y$), which makes it a strict saddle point.

---

## Update Rules

### 1. Standard Newton vs. Saddle-Free Newton (SFN)
- **Standard Newton**: 
  $$z_{t+1} = z_t - H^{-1} g$$
  Near the saddle point, Newton's method uses curvature to jump directly *towards* the critical point. Since the gradient at the saddle point is exactly zero, it gets trapped there forever.
- **Saddle-Free Newton (SFN)**:
  $$z_{t+1} = z_t - |H|^{-1} g$$
  SFN takes the absolute value of the Hessian eigenvalues $|H| = Q |\Lambda| Q^T$ (meaning $|H| = \text{diag}(2, |-1 + 3y^2|)$). By making the curvature matrix positive-definite, SFN converts negative curvature directions into active descent steps, ensuring rapid and exponential escape.

### 2. Gradient Descent (GD) vs. Momentum (GDM)
- **GD**: Escapes slowly because the gradient along the negative curvature direction ($y$) is very small near the saddle point, creating a long stalling plateau (the flat valley).
- **GDM**: Accumulates velocity along $y$, allowing it to glide out of the flat valley much faster.

---

## Empirical Results

When initialized extremely close to the saddle point at $x_0 = (0.1, 0.001)$:
- **Standard Newton** converges to the saddle point $(0,0)$ in exactly $1$ step and remains trapped ($g = 0$) forever.
- **GD** gets stuck in the flat valley, plateauing for over 100 steps before finally escaping to the local minimum $(0, 1)$.
- **GDM** builds momentum and escapes the plateau in about 40 steps.
- **SFN** exponentially pushes away from the saddle point from step 1, arriving at the local minimum $(0, 1)$ almost instantly.

Visualizations of the trajectories, surface wireframes, loss paths, and gradient norms are saved in the `plots/` folder.

---

## Project Structure

- `experiment.py`: Main Python implementation of the landscape, gradients, Hessians, and the optimization routines.
- `plot.py`: Visualization code generating 2D contours, 3D surface wires, loss, and gradient norm curves.
- `run.sh`: Shell runner script leveraging the `rl_sim` conda environment.
- `optimization_results.json`: Saved raw simulation logs.
- `plots/`: Output folder for PNG plots.

---

## How to Run

Execute the runner script:
```bash
chmod +x run.sh
./run.sh
```
This runs the simulation, saves the coordinates, and generates four plots in `plots/`:
1. `contour_trajectory.png`: 2D trajectories overlaid on contour levels.
2. `surface_trajectory_3d.png`: 3D trajectories overlaid on the wireframe landscape.
3. `loss_vs_iteration.png`: The loss values over time.
4. `gradient_norm_vs_iteration.png`: Gradient norms over time.
