import numpy as np

# apply logarithmic transformation to a skewed feature to reduce its skewness
df["bytes_transferred"] = np.log1p(df["bytes_transferred"]) # add 1 to avoid log(0)