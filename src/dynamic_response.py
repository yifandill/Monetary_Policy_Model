import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
from jax import jit, random
from model import MonetaryModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from config import benchmark_dict
    model = MonetaryModel(**benchmark_dict)     # Initialize model
    P = model.solve_P()    
    #P = model.solve_P(kappa=0.5, , rho_r=0.8, rho_u=0.8phi_pi=5, phi_x=3)

    T = 30      # Number of periods
    prev_shock = jnp.zeros(3, dtype=jnp.float32)     # Example usage with a zero initial shock

    # One std dev change of different shock at t=0
    t = 0
    shock_index = 0    # 0 : u, 1 : r, 2 : nu.
    std_dev_change = 1.0
    changed_shock = jnp.zeros(3, dtype=jnp.float32).at[shock_index].set(std_dev_change)
    shocks = jnp.zeros((T, 3), dtype=jnp.float32).at[t].set(changed_shock)  
    #shocks = jnp.zeros((T, 3), dtype=jnp.float32).at[t].set(jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32))

    # Simulate for T periods
    shock_series = []
    measurement_series = []

    for t in range(T):
        current_shock = model.transition(prev_shock, shock_override=shocks[t])
        #current_shock = model.transition(prev_shock, shock_override=shocks[t],rho_r=0.8, rho_u=0.8)
        measurement = model.measurement(P, current_shock)
        shock_series.append(current_shock)
        measurement_series.append(measurement)
        prev_shock = current_shock

    shock_series = jnp.array(shock_series)
    measurement_series = jnp.array(measurement_series)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    labels = [r'$\pi_{t+i}$', r'$x_{t+i}$', r'$i_{t+i}$']
    shocks_name = [r'$ε_{u}$', r'$ε_{r}$', r'$ε_{v}$']
    colors = ['orangered', 'violet', 'dodgerblue']

    for i in range(3):
        axes[i].plot(range(T), measurement_series[:, i], marker='o', linestyle='-', color=colors[i])
        axes[i].set_title(f'Response of {labels[i]} to a One Std Dev Change of {shocks_name[shock_index]}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(labels[i])
        axes[i].grid()

    plt.tight_layout()
    plt.show()


