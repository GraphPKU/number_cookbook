import sys
import os

# Get the absolute path to the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')

# Add 'src' directory to the Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)