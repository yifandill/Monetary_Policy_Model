import os
import sys
from pathlib import Path
from warnings import filterwarnings
import platform

filterwarnings('ignore')
sys.path.append(Path('.'))

# Check conditions for setting JAX to use CPU only
def configure_jax_platform():
    # Case 1: Check if we're on iOS (which doesn't support GPU acceleration for JAX)
    if platform.system() == "Darwin" and platform.machine().startswith(("arm", "iPhone")):
        print("iOS/macOS ARM device detected. Setting JAX to use CPU only.")
        os.environ["JAX_PLATFORMS"] = "cpu"
        return True
        
    # Case 2: Try to detect if CUDA/GPU is available
    try:
        # We'll try to import jax first without setting the platform
        import jax
        devices = jax.devices()
        
        # If no GPU devices found, set to CPU
        if all(d.platform == 'cpu' for d in devices):
            print("No GPU devices detected. Setting JAX to use CPU only.")
            os.environ["JAX_PLATFORMS"] = "cpu"
            return True
            
        # GPU available, no need to change anything
        print(f"GPU device(s) available: {[str(d) for d in devices]}")
        return False
        
    except (ImportError, Exception) as e:
        # If importing jax fails or any other error occurs, default to CPU
        print(f"Error checking JAX devices: {e}. Setting JAX to use CPU only.")
        os.environ["JAX_PLATFORMS"] = "cpu"
        return True

# Run the configuration function
using_cpu_only = configure_jax_platform()

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

benchmark_dict_opt = {
    'beta': 0.99,
    'sigma': 1/6,
    'kappa': 0.024,
    'rho_r': 0.35,
    'rho_u': 0.35,
    'sigma_r': 13.8,
    'sigma_u': 0.17,
    'lambda_x': 0.048,
    'lambda_i': 0.236
}
