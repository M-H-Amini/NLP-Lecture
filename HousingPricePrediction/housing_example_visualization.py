##  Generate a beautiful plot of a dataset for housing prices.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the inputs and outputs
X = np.array([50, 90, 100, 120, 150, 230, 300, 310, 400, 500])
y = np.array([400, 700, 800, 850, 900, 1000, 1200, 1500, 2000, 3000])

sns.set(font_scale=1.5)
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='red', s=70)
plt.xlabel('X (Size)')
plt.ylabel('Y (Price)')
plt.title('Housing Prices')
plt.tight_layout()
plt.savefig('housing.png', bbox_inches='tight')
plt.show()