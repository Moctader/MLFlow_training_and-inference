import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate sample data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=100)

# Sort the data
sorted_data = np.sort(data)

# Calculate ECDF values
y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# Calculate theoretical CDF values
mean, std = np.mean(data), np.std(data)
theoretical_cdf = norm.cdf(sorted_data, loc=mean, scale=std)

# Plot ECDF and theoretical CDF
plt.figure(figsize=(10, 6))
plt.step(sorted_data, y, where='post', label='ECDF', color='blue')
plt.plot(sorted_data, theoretical_cdf, label='Theoretical CDF (Gaussian)', color='red', linestyle='--')
plt.xlabel('Value', fontsize=14)
plt.ylabel('CDF', fontsize=14)
plt.title('Empirical Cumulative Distribution Function (ECDF) vs Theoretical Gaussian CDF', fontsize=16)
plt.legend(fontsize=12)
plt.show()