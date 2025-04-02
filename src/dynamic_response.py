import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
from jax import jit, random
from model import MonetaryModel
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initialize model
    from config import benchmark_dict
    model = MonetaryModel(**benchmark_dict)
    P = model.solve_P()

    # Example usage with a zero initial shock
    T = 30  # Number of periods
    prev_shock = jnp.zeros(3, dtype=jnp.float32)

    # One std dev change of different shock at t=0
    t = 0
    shock_index = 0     # 0 : u, 1 : r, 2 : nu.
    std_dev_change = 1.0
    changed_shock = jnp.zeros(3, dtype=jnp.float32).at[shock_index].set(std_dev_change)
    shocks = jnp.zeros((T, 3), dtype=jnp.float32).at[t].set(changed_shock)  
    
    shock_series = []
    measurement_series = []

    # Simulate for T periods
    for t in range(T):
        current_shock = model.transition(prev_shock, shock_override=shocks[t])
        measurement = model.measurement(P, current_shock)
        shock_series.append(current_shock)
        measurement_series.append(measurement)
        prev_shock = current_shock

    # Convert results to numpy for plotting
    shock_series = jnp.array(shock_series)
    measurement_series = jnp.array(measurement_series)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    labels = [r'$\pi_{t+i}$', r'$x_{t+i}$', r'$i_{t+i}$']
    shocks_name = [r'$ε_{u}$', r'$ε_{r}$', r'$ε_{v}$']

    for i in range(3):
        axes[i].plot(range(T), measurement_series[:, i], marker='o', linestyle='-')
        axes[i].set_title(f'Response of {labels[i]} to a One Std Dev Change of {shocks_name[shock_index]}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(labels[i])
        axes[i].grid()

    plt.tight_layout()
    plt.show()


