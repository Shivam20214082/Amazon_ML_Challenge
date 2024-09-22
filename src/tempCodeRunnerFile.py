import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import utils

print(f"Current sys.path: {sys.path}")

try:
    from utils import predictor
    print("predictor function:", predictor("link", "entity"))
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
