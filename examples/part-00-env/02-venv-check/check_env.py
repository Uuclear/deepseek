import sys
print("executable:", sys.executable)
try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError:
    print("numpy: not installed (run pip install numpy)")
