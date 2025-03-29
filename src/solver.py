import numpy as np


def solve_P(beta, sigma, kappa, phi_pi, phi_x, rho_u, rho_r, rho_nu, sigma_r, sigma_u, sigma_nu):
    """
    Solve for the matrix P given a dictionary of parameters.
    Returns the 3x3 matrix P.
    """
    # Build the 9x9 matrix A
    A = np.array([
        [1 - beta * rho_u,     -kappa,            0,                 0,                0,     0,                 0,     0,     0],
        [0,                    0,                 0,                 1 - beta * rho_r, -kappa,0,                 0,     0,     0],
        [0,                    0,                 0,                 0,                0,     0, 1 - beta * rho_nu, -kappa, 0],
        
        [sigma * rho_u,        rho_u - 1,        -sigma,             0,                0,     0,                 0,     0,     0],
        [0,                    0,                 0,                 sigma * rho_r,    rho_r - 1, -sigma,        0,     0,     0],
        [0,                    0,                 0,                 0,                0,     0, sigma * rho_nu, rho_nu - 1, -sigma],
        
        [phi_pi,               phi_x,            -1,                 0,                0,     0,                 0,     0,     0],
        [0,                    0,                 0,                 phi_pi,           phi_x,  -1,               0,     0,     0],
        [0,                    0,                 0,                 0,                0,     0, phi_pi,         phi_x,  -1]
    ])

    # Build the 9x1 vector b
    b = np.array([1, 0, 0, 0, -sigma, 0, 0, 0, -1], dtype=float)

    # Solve the system A x = b
    x = np.linalg.solve(A, b)

    # Reshape the solution x into the 3x3 matrix P. Use order='F' so that the first 3 elements of x become the first column of P (consistent with "vec" stacking columns).
    P = x.reshape(3, 3, order='F')

    return P


if __name__ == "__main__":
    from config import benchmark_dict
    # Benchmark parameters
    P = solve_P(**benchmark_dict)
    print(P)
