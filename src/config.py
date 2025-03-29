import os
import sys
from pathlib import Path
from warnings import filterwarnings

filterwarnings('ignore')
sys.path.append(Path('.'))

script_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_dir, '..', 'results')
