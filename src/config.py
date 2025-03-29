import os
import sys
from pathlib import Path
from warnings import filterwarnings

filterwarnings('ignore')
sys.path.append(Path('.'))

script_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_dir, '..', 'results')

benchmark_dict = {
    'beta': 0.99,
    'sigma': 1/6,
    'kappa': 0.024,
    'phi_pi': 1.5,
    'phi_x': 0.5,
    'rho_r': 0.35,
    'rho_u': 0.35,
    'rho_nu': 0.35,
    'sigma_r': 3.7,
    'sigma_u': 0.4,
    'sigma_nu': 1,
    'seed': 0
}
