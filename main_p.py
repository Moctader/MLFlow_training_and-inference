import numpy as np
from scipy.stats import ks_2samp

# Generate sample data
np.random.seed(0)
data1 = np.random.normal(loc=0, scale=1, size=100)
data2 = np.random.normal(loc=1, scale=1, size=100)  # Different mean for demonstration

# Sort the data
sorted_data1 = np.sort(data1)
sorted_data2 = np.sort(data2)

# Calculate ECDF values
ecdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
ecdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)

# Compute the KS statistic
D = np.max(np.abs(ecdf1 - ecdf2))

# Calculate the p-value using the asymptotic distribution of the KS statistic
n1 = len(data1)
n2 = len(data2)
en = np.sqrt(n1 * n2 / (n1 + n2))
p_value = ks_2samp(data1, data2).pvalue

print(f"KS statistic: {D}")
print(f"p-value: {p_value}")