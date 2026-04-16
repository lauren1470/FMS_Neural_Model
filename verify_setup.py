from brian2 import *
import brian2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn
import pandas as pd

print("FMS NEURAL MODEL - ENVIRONMENT VERIFICATION")

print("\n✓ Core dependencies installed:")
print(f"  Brian2:        {brian2.__version__}")
print(f"  NumPy:         {np.__version__}")
print(f"  Matplotlib:    {plt.matplotlib.__version__}")
print(f"  Scikit-learn:  {sklearn.__version__}")
print(f"  Pandas:        {pd.__version__}")

print("\n✓ Machine learning classifiers available:")
print(f"  Random Forest: {RandomForestClassifier.__name__}")
print(f"  SVM:           {SVC.__name__}")

print("\n✓ Brian2 code generation target:")
print(f"  {prefs.codegen.target}")

print("READY TO BUILD FMS MODEL")