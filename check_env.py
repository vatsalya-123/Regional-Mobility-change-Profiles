import sys
import pandas as pd, numpy as np, scipy, sklearn
import matplotlib, seaborn, plotly, pyarrow, umap, streamlit, yaml

print("Python :", sys.version.split()[0])
for lib in [pd, np, scipy, sklearn, matplotlib, seaborn, plotly, pyarrow, streamlit]:
    print(f"{lib.__name__:12}", lib.__version__)
print("umap-learn  :", umap.__version__)
print("\nAll good. Environment ready.")