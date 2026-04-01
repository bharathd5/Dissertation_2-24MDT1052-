import sys
import os

print("Current working directory:", os.getcwd())
print("\nPython path:")
for p in sys.path:
    print(" ", p)

print("\nFiles in current directory:")
for f in os.listdir("."):
    if f.endswith(".py"):
        print(" ", f)