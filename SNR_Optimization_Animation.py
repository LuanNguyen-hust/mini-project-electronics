import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.optimize import differential_evolution

# Specify the path to ffmpeg executable (adjust based on your installation)
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\giyua\Downloads\ffmpeg-2025-05-26-git-43a69886b2-essentials_build\ffmpeg-2025-05-26-git-43a69886b2-essentials_build\bin\ffmpeg.exe'

# Define the SNR function
def snr(Z_L, g, Z_RT, S_I_T, N_na, k, T, Z_R):
    denom_sum = Z_R + Z_L
    small_mask = np.abs(denom_sum) < 1e-12  # Mask where denom is too small

    # Avoid divide-by-zero by substituting with zeros where denom_sum is too small
    safe_Z_L_over_sum = np.where(small_mask, 0, Z_L / denom_sum)
    safe_Z_R_over_sum = np.where(small_mask, 0, Z_R / denom_sum)

    numerator = g**2 * np.abs(safe_Z_L_over_sum)**2 * np.abs(Z_RT)**2 * S_I_T
    denominator = N_na + g**2 * np.abs(safe_Z_R_over_sum)**2 * 2 * k * T * np.real(Z_L)

    # Final SNR calculation (avoid division by zero in final denominator too)
    return np.where(denominator == 0, np.inf, numerator / denominator)

# Define the objective function to minimize (-SNR)
# x = [Re_Z_L, Im_Z_L, Re_Z_R, Im_Z_R]
def objective(x, g, Z_RT, S_I_T, N_na, k, T, Z_R):
    Re_Z_L, Im_Z_L = x
    Z_L = Re_Z_L + 1j * Im_Z_L

    return -snr(Z_L, g, Z_RT, S_I_T, N_na, k, T, Z_R)

# Set fixed parameters
g = 1
Z_T = 50 + 50j  # Transmitter antenna self-impedance (for reference, not used directly)
Z_RT = 50 + 1j  # Mutual impedance (assumed)
S_I_T = 1
N_na = 1e-9
k = 1.38e-23
T = 300
Z_R = 50 + 200j

# Define bounds for optimization
# [Re_Z_L, Im_Z_L, Re_Z_R, Im_Z_R]
# Re_Z_L and Re_Z_R must be >= 0 (physical constraint for passive impedances)
bounds = [(0.1, 1000), (-1000, 1000)]

# Lists to store best parameters and population at each iteration
best_params_list = []
population_history = []
fitness_history = []

# Modified callback function to record population and fitness
def callback(xk, convergence):
    best_params_list.append(xk)
    # Note: The callback doesn't give us access to the full population
    # We'll need to use a custom optimization wrapper

# Custom wrapper class to capture population
class PopulationTracker:
    def __init__(self):
        self.populations = []
        self.best_candidates = []
        self.fitnesses = []
    
    def __call__(self, func, bounds, args=(), **kwargs):
        # Store original callback
        original_callback = kwargs.get('callback', None)
        
        def new_callback(xk, convergence):
            self.best_candidates.append(xk.copy())
            if original_callback:
                original_callback(xk, convergence)
        
        kwargs['callback'] = new_callback
        
        # We'll monkey patch the differential evolution to capture populations
        # This is a bit hacky but necessary since scipy doesn't expose population
        original_differential_evolution = differential_evolution
        
        def patched_de(func, bounds, **de_kwargs):
            # Create a wrapper that captures intermediate results
            result = original_differential_evolution(func, bounds, **de_kwargs)
            return result
        
        return patched_de(func, bounds, **kwargs)

# Since we can't easily access the internal population from scipy's differential_evolution,
# let's create a simplified version that captures what we need
def simple_differential_evolution_with_tracking(bounds, popsize=50, maxiter=100):
    """Simplified DE that tracks population evolution"""
    n_params = len(bounds)
    
    # Initialize population
    pop = np.random.rand(popsize, n_params)
    for i in range(n_params):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])
    
    population_history.clear()
    best_params_list.clear()
    fitness_history.clear()
    
    for generation in range(maxiter):
        # Evaluate fitness for all individuals
        fitness = np.array([objective(ind, g, Z_RT, S_I_T, N_na, k, T, Z_R) for ind in pop])
        
        # Store current population and fitness
        population_history.append(pop.copy())
        fitness_history.append(fitness.copy())
        
        # Find best individual
        best_idx = np.argmin(fitness)
        best_params_list.append(pop[best_idx].copy())
        
        # Simple DE mutation and crossover (simplified version)
        new_pop = pop.copy()
        F = 0.8  # mutation factor
        CR = 0.7  # crossover probability
        
        for i in range(popsize):
            # Select three random individuals (different from current)
            candidates = list(range(popsize))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Ensure bounds
            for j in range(n_params):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n_params):
                if np.random.rand() < CR or j == np.random.randint(n_params):
                    trial[j] = mutant[j]
            
            # Selection
            trial_fitness = objective(trial, g, Z_RT, S_I_T, N_na, k, T, Z_R)
            if trial_fitness < fitness[i]:
                new_pop[i] = trial
        
        pop = new_pop
        
        # Early stopping if converged
        if generation > 10 and np.std(fitness) < 1e-6:
            break
    
    # Final evaluation
    final_fitness = np.array([objective(ind, g, Z_RT, S_I_T, N_na, k, T, Z_R) for ind in pop])
    best_idx = np.argmin(final_fitness)
    
    return pop[best_idx], -final_fitness[best_idx]

# Run optimization with population tracking
print("Running optimization with population tracking...")
result_x, result_fun = simple_differential_evolution_with_tracking(bounds, popsize=30, maxiter=50)

# Extract optimal results
Re_Z_L_opt, Im_Z_L_opt = result_x
Z_L_opt = Re_Z_L_opt + 1j * Im_Z_L_opt
snr_opt = result_fun
print(f"Optimal Z_L: {Z_L_opt}")
print(f"Optimal Z_R: {Z_R}")
print(f"Optimal SNR: {snr_opt}")

# Compare with open-circuit and matched cases (using optimal Z_R)
Z_L_oc = 1e6 + 0j
Z_L_match = np.conj(Z_R)
snr_oc = snr(Z_L_oc, g, Z_RT, S_I_T, N_na, k, T, Z_R)
snr_match = snr(Z_L_match, g, Z_RT, S_I_T, N_na, k, T, Z_R)
print(f"SNR (Open-Circuit with Optimal Z_R): {snr_oc}")
print(f"SNR (Matched with Optimal Z_R): {snr_match}")

# Set up the animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 1000)
ax.set_ylim(-1000, 1000)
ax.set_xlabel('Re(Z_L)')
ax.set_ylabel('Im(Z_L)')
ax.set_title('Differential Evolution Optimization with Population Visualization')

# Initialize plot elements
contour = None
colorbar = None
population_scatter = None
best_marker = None
text = None

# Animation functions
def init_animation():
    global contour, colorbar, population_scatter, best_marker, text, fig, ax
    
    # Define Z_L grid for contour plot
    re_z_l_vals = np.linspace(0, 1000, 100)
    im_z_l_vals = np.linspace(-1000, 1000, 100)
    Re_Z_L_mesh, Im_Z_L_mesh = np.meshgrid(re_z_l_vals, im_z_l_vals)
    Z_L_grid_mesh = Re_Z_L_mesh + 1j * Im_Z_L_mesh

    snr_grid_init = 10 * np.log10(snr(Z_L_grid_mesh, g, Z_RT, S_I_T, N_na, k, T, Z_R))

    # Clear the axes completely
    ax.clear()

    # Plot initial contour
    contour = ax.contourf(Re_Z_L_mesh, Im_Z_L_mesh, snr_grid_init, levels=100, cmap='viridis', extend='both')

    # Set up axes labels and title
    ax.set_xlabel('Re(Z_L)')
    ax.set_ylabel('Im(Z_L)')
    ax.set_title('Differential Evolution Optimization with Population Visualization')
    ax.set_xlim(0, 1000)
    ax.set_ylim(-1000, 1000)

    # Create colorbar
    if colorbar is None:
        colorbar = fig.colorbar(contour, ax=ax, label='SNR (dB)')

    # Initialize empty plots
    population_scatter = ax.scatter([], [], c=[], s=30, alpha=0.6, cmap='coolwarm', 
                                  edgecolors='black', linewidth=0.5, label='Population')
    best_marker, = ax.plot([], [], 'r*', markersize=10, label='Best Candidate')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Initialize text
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return []

def update_animation(frame):
    global contour, colorbar, population_scatter, best_marker, text, ax
    
    if frame >= len(population_history):
        return []
    
    # Get current population and best candidate
    current_pop = population_history[frame]
    current_fitness = fitness_history[frame]
    best_candidate = best_params_list[frame]
    
    # Recompute SNR grid
    re_z_l_vals = np.linspace(0, 1000, 100)
    im_z_l_vals = np.linspace(-1000, 1000, 100)
    Re_Z_L_mesh, Im_Z_L_mesh = np.meshgrid(re_z_l_vals, im_z_l_vals)
    Z_L_grid_mesh = Re_Z_L_mesh + 1j * Im_Z_L_mesh

    snr_grid_current = 10 * np.log10(snr(Z_L_grid_mesh, g, Z_RT, S_I_T, N_na, k, T, Z_R))

    # Clear and redraw
    ax.clear()
    
    # Create contour plot
    contour = ax.contourf(Re_Z_L_mesh, Im_Z_L_mesh, snr_grid_current, levels=100, cmap='viridis', extend='both')
    
    # Set up axes
    ax.set_xlabel('Re(Z_L)')
    ax.set_ylabel('Im(Z_L)')
    ax.set_title('Differential Evolution Optimization with Population Visualization')
    ax.set_xlim(0, 1000)
    ax.set_ylim(-1000, 1000)

    # Update colorbar
    if colorbar is not None:
        colorbar.update_normal(contour)
    else:
        colorbar = fig.colorbar(contour, ax=ax, label='SNR (dB)')

    # Plot population points
    pop_re = current_pop[:, 0]  # Real parts
    pop_im = current_pop[:, 1]  # Imaginary parts
    
    # Color population points by fitness (blue=good, red=bad)
    normalized_fitness = (current_fitness - current_fitness.min()) / (current_fitness.max() - current_fitness.min() + 1e-10)
    
    population_scatter = ax.scatter(pop_re, pop_im, c=normalized_fitness, s=50, alpha=0.7, 
                                  cmap='coolwarm_r', edgecolors='black', linewidth=0.5, 
                                  label=f'Population (n={len(current_pop)})')
    
    # Plot best candidate
    best_marker, = ax.plot([best_candidate[0]], [best_candidate[1]], 'r*', 
                          markersize=10, label='Best Candidate')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Calculate SNR at best point
    Z_L_best = best_candidate[0] + 1j * best_candidate[1]
    snr_value_at_best = 10 * np.log10(snr(Z_L_best, g, Z_RT, S_I_T, N_na, k, T, Z_R))
    
    # Update text
    text = ax.text(0.02, 0.98, 
                   f"Generation: {frame}\n"
                   f"Population Size: {len(current_pop)}\n"
                   f"Best SNR: {snr_value_at_best:.2f} dB\n"
                   f"Best Z_L: {best_candidate[0]:.1f} + {best_candidate[1]:.1f}j\n"
                   f"Z_R: {Z_R.real:.1f} + {Z_R.imag:.1f}j", 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    return []

# Create and save animation
print(f"Creating animation with {len(population_history)} frames...")
try:
    writer = FFMpegWriter(fps=5, bitrate=8000)  # Slower fps to see population evolution
    ani = FuncAnimation(fig, update_animation, frames=len(population_history), 
                       init_func=init_animation, blit=False, repeat=True)
    ani.save('optimization_with_population.mp4', writer=writer)
    print("Animation saved as 'optimization_with_population.mp4'")
except Exception as e:
    print(f"Error saving animation as MP4: {e}")
    print("Falling back to GIF format with Pillow writer")
    try:
        ani.save('optimization_with_population.gif', writer='pillow', fps=3)
        print("Animation saved as 'optimization_with_population.gif'")
    except Exception as e2:
        print(f"Error saving as GIF: {e2}")

# Close the plot to free memory
plt.close()