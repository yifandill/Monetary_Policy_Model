import functools
import numpy as np
import random as py_random

def _solve_P(
    beta: float,
    sigma: float,
    kappa: float,
    phi_pi: float,
    phi_x: float,
    rho_u: float,
    rho_r: float,
    rho_nu: float,
) -> np.ndarray:
    # 构建9x9系数矩阵A
    A = np.array([
        [1 - beta * rho_u,  -kappa,              0,                   0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   1 - beta * rho_r, -kappa, 0,           0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, 1 - beta * rho_nu, -kappa, 0],

        [sigma * rho_u,     rho_u - 1,          -sigma,               0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   sigma * rho_r,    rho_r - 1, -sigma,   0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, sigma * rho_nu, rho_nu - 1, -sigma],

        [phi_pi,            phi_x,              -1,                   0,                0,   0,               0,    0,   0],
        [0,                 0,                   0,                   phi_pi,           phi_x, -1,            0,    0,   0],
        [0,                 0,                   0,                   0,                0,   0, phi_pi,        phi_x, -1]
    ], dtype=np.float32)

    # 构建9x1向量b
    b = np.array([1, 0, 0, 0, -sigma, 0, 0, 0, -1], dtype=np.float32)

    # 求解线性方程组 A @ x = b
    x = np.linalg.solve(A, b)

    # 将x重塑为3x3矩阵
    P = x.reshape((3, 3), order='F')
    return P

class MonetaryModel:


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
        # 设置随机数种子
        np.random.seed(seed)
        py_random.seed(seed)

    def solve_P(self, 
                kappa: float = None, 
                phi_pi: float = None, 
                phi_x: float = None,
                rho_u: float = None,
                rho_r: float = None,
                ) -> np.ndarray:
        """
        通过调用独立函数来求解3x3矩阵P。
        """
        return _solve_P(
            self.beta,
            self.sigma,
            kappa if kappa is not None else self.kappa,
            phi_pi if phi_pi is not None else self.phi_pi,
            phi_x if phi_x is not None else self.phi_x,
            rho_u if rho_u is not None else self.rho_u,
            rho_r if rho_r is not None else self.rho_r,
            self.rho_nu
        )

    def transition(self, 
                   prev_shock: np.ndarray, 
                   shock_override: np.ndarray = None,
                   rho_u: float = None,
                   rho_r: float = None
                   ) -> np.ndarray:

        F = np.diag([
            rho_u if rho_u is not None else self.rho_u,
            rho_r if rho_r is not None else self.rho_r,
            self.rho_nu
        ]).astype(np.float32)
        
        G = np.diag([self.sigma_u, self.sigma_r, self.sigma_nu]).astype(np.float32)
                
        if shock_override is None:
            # 使用NumPy生成随机噪声
            noise = np.random.normal(0, 1, 3).astype(np.float32)
        else:
            noise = shock_override  # 手动覆盖冲击
        
        return F @ prev_shock + G @ noise

    def measurement(self, P: np.ndarray, shock: np.ndarray) -> np.ndarray:
        return P @ shock


if __name__ == "__main__":
    from config import benchmark_dict
    # 用基准参数实例化MonetaryModel
    model = MonetaryModel(**benchmark_dict)
    
    # 求解P
    P = model.solve_P()
    print("矩阵 P:\n", P)

    # 使用零初始冲击的示例用法
    prev_shock = np.zeros(3, dtype=np.float32)
    current_shock = model.transition(prev_shock)
    measurement_out = model.measurement(P, current_shock)

    # 打印结果
    print("下一个冲击:\n", current_shock)
    print("测量结果:\n", measurement_out)
    prev_shock = jnp.zeros(3, dtype=jnp.float32)
    current_shock = model.transition(prev_shock)
    measurement_out = model.measurement(P, current_shock)

    # Print results
    print("Next shock:\n", current_shock)
    print("Measurement:\n", measurement_out)
