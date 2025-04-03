import functools
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
from jax import jit, random

def _solve_P_jit(
    beta: float,
    sigma: float,
    kappa: float,
    phi_pi: float,
    phi_x: float,
    rho_u: float,
    rho_r: float,
    rho_nu: float,
) -> jnp.ndarray:
    """
    A standalone JIT-compiled function to solve for the 3x3 matrix P.
    """
    # Construct the 9x9 coefficient matrix A
    A = jnp.array([
        [1 - beta * rho_u,  -kappa,              0,                   0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   1 - beta * rho_r, -kappa, 0,           0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, 1 - beta * rho_nu, -kappa, 0],

        [sigma * rho_u,     rho_u - 1,          -sigma,               0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   sigma * rho_r,    rho_r - 1, -sigma,   0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, sigma * rho_nu, rho_nu - 1, -sigma],

        [phi_pi,            phi_x,              -1,                   0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   phi_pi,           phi_x, -1,            0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, phi_pi,        phi_x, -1]
    ], dtype=jnp.float32)

    # Construct the 9x1 vector b
    b = jnp.array([1, 0, 0, 0, -sigma, 0, 0, 0, -1], dtype=jnp.float32)

    # Solve the linear system A @ x = b
    x = jnp.linalg.solve(A, b)

    # Reshape x into a 3x3 matrix
    P = x.reshape((3, 3), order='F')
    return P

# JIT-compile the standalone solver function
_solve_P_jit = jit(_solve_P_jit)

class MonetaryModel:
    """
    A class encapsulating the functionality of solving matrix P, handling state transitions,
    and computing measurement outputs. Parameters are stored as attributes at initialization.
    """

    def __init__(
        self,
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
        sigma_nu: float,
        seed: int = 0
    ):
        self.beta = beta
        self.sigma = sigma
        self.kappa = kappa
        self.phi_pi = phi_pi
        self.phi_x = phi_x
        self.rho_u = rho_u
        self.rho_r = rho_r
        self.rho_nu = rho_nu
        self.sigma_r = sigma_r
        self.sigma_u = sigma_u
        self.sigma_nu = sigma_nu
        self.seed = seed

    def solve_P(self, 
                kappa: float = None, 
                phi_pi: float = None, 
                phi_x: float = None,
                rho_u: float = None,
                rho_r: float = None,
                ) -> jnp.ndarray:
        """
        Solve for the 3x3 matrix P by calling the standalone JIT-compiled function.
        """
        return _solve_P_jit(
            self.beta,
            self.sigma,
            kappa if kappa is not None else self.kappa,
            phi_pi if phi_pi is not None else self.phi_pi,
            phi_x if phi_x is not None else self.phi_x,
            rho_u if rho_u is not None else self.rho_u,
            rho_r if rho_r is not None else self.rho_r,
            self.rho_nu
        )

    @functools.partial(jit, static_argnames=("self",))
    def transition(self, 
                   prev_shock: jnp.ndarray, 
                   shock_override: jnp.ndarray = None,
                   rho_u: float = None,
                   rho_r: float = None
                   ) -> jnp.ndarray:
        """
        Compute the next shock state using the transition matrix and random noise.
        
        Args:
            prev_shock (jnp.ndarray): The previous shock vector.
            rho_u (float): Optional override for rho_u.
            rho_r (float): Optional override for rho_r.
        
        Returns:
            jnp.ndarray: The current shock vector after transition.
        """
        F = jnp.diag(jnp.array([
            rho_u if rho_u is not None else self.rho_u,
            rho_r if rho_r is not None else self.rho_r,
            self.rho_nu
        ], dtype=jnp.float32))
        G = jnp.diag(jnp.array([self.sigma_u, self.sigma_r, self.sigma_nu], dtype=jnp.float32))
                
        if shock_override is None:
            key = random.PRNGKey(self.seed)
            noise = random.normal(key, shape=(3,))
        else:
            noise = shock_override  # Manually override the shock
        
        return F @ prev_shock + G @ noise

    @functools.partial(jit, static_argnames=("self",))
    def measurement(self, P: jnp.ndarray, shock: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the measurement output given P and the current shock.

        Args:
            P (jnp.ndarray): The 3x3 matrix.
            shock (jnp.ndarray): The shock vector.

        Returns:
            jnp.ndarray: The measurement result.
        """
        return P @ shock


if __name__ == "__main__":
    from config import benchmark_dict
    # Instantiate the MonetaryModel with benchmark parameters
    model = MonetaryModel(**benchmark_dict)
    
    # Solve for P
    P = model.solve_P()
    print("Matrix P:\n", P)

    # Example usage with a zero initial shock
    prev_shock = jnp.zeros(3, dtype=jnp.float32)
    current_shock = model.transition(prev_shock)
    measurement_out = model.measurement(P, current_shock)

    # Print results
    print("Next shock:\n", current_shock)
    print("Measurement:\n", measurement_out)
