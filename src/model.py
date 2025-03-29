import jax.numpy as jnp
from jax import jit
from scipy.stats import norm

@jit
def solve_P(
    beta: float,
    sigma: float,
    kappa: float,
    phi_pi: float,
    phi_x: float,
    rho_u: float,
    rho_r: float,
    rho_nu: float,
    sigma_r: float,
    sigma_u: float,
    sigma_nu: float
) -> jnp.ndarray:
    """
    Solve for the 3x3 matrix P using JAX for potential acceleration through JIT.

    Returns:
        jnp.ndarray: A 3x3 matrix P.
    """
    # Construct the 9x9 coefficient matrix A using jnp
    A = jnp.array([
        [1 - beta * rho_u,     -kappa,             0,                 0,               0,     0,                 0,     0,     0],
        [0,                    0,                  0,                 1 - beta * rho_r,-kappa,0,                 0,     0,     0],
        [0,                    0,                  0,                 0,               0,     0, 1 - beta * rho_nu, -kappa, 0],
        
        [sigma * rho_u,        rho_u - 1,         -sigma,             0,               0,     0,                 0,     0,     0],
        [0,                    0,                  0,                 sigma * rho_r,   rho_r - 1, -sigma,        0,     0,     0],
        [0,                    0,                  0,                 0,               0,     0, sigma * rho_nu,  rho_nu - 1, -sigma],
        
        [phi_pi,               phi_x,             -1,                 0,               0,     0,                 0,     0,     0],
        [0,                    0,                  0,                 phi_pi,          phi_x,  -1,               0,     0,     0],
        [0,                    0,                  0,                 0,               0,     0, phi_pi,         phi_x,  -1]
    ], dtype=jnp.float32)

    # Construct the 9x1 vector b using jnp
    b = jnp.array([1, 0, 0, 0, -sigma, 0, 0, 0, -1], dtype=jnp.float32)

    # Solve the linear system A @ x = b
    x = jnp.linalg.solve(A, b)

    # Reshape x into a 3x3 matrix (column-major order)
    P = x.reshape((3, 3), order='F')
    return P


def transition(
    prev_shock: jnp.ndarray,
    rho_u: float,
    rho_r: float,
    rho_nu: float,
    sigma_r: float,
    sigma_u: float,
    sigma_nu: float
) -> jnp.ndarray:
    # Transition matrix
    F = jnp.diag([rho_u, rho_r, rho_nu])

    # Shock/loading matrix
    G = jnp.diag([sigma_u, sigma_r, sigma_nu])

    noise = norm.rvs(size=3)
    current_shock = F @ prev_shock + G @ noise

    return current_shock


def measurement(P: jnp.ndarray, shock: jnp.ndarray) -> jnp.ndarray:
    return P @ shock


if __name__ == "__main__":
    from config import benchmark_dict
    # Benchmark parameters
    P = solve_P(**benchmark_dict)
    print(P)
