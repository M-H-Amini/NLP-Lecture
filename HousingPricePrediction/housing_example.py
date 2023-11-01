##  Generate a beautiful plot of a dataset for housing prices.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the inputs and outputs
X = np.array([[50, 90, 100, 120, 150, 230, 300, 310, 400, 500]]).T
y = np.array([[400, 700, 800, 850, 900, 1000, 1200, 1500, 2000, 3000]]).T

class MHNeuron:
    def __init__(self, w=None, b=None):
        self.w = w if w is not None else np.random.randn()
        self.b = b if b is not None else np.random.randn()

    def forward(self, x):
        return self.w*x + self.b
    
    def fit(self, X, y, epochs=10, lr=0.1):
        m = len(X)
        for epoch in range(epochs):
            y_hat = self.forward(X)
            print('y_hat shape:', y_hat.shape)
            print('y_hat:', y_hat)
            error = y - y_hat
            print('Error shape:', error.shape)
            print('Error:', error)
            print('X shape:', X.shape)
            print(error.T.dot(X).shape)
            self.w += float(lr / m * error.T.dot(X)[0, 0])
            self.b += float(lr / m * error[0, 0])

def visualize(X, y, model):
    X_min, X_max = X.min() - 10, X.max() + 10
    y_hat = model.forward(np.linspace(X_min, X_max, 100)[np.newaxis, :].T)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='red', s=70)
    plt.plot(np.linspace(X_min, X_max, 100), y_hat, color='green', linewidth=3)
    plt.xlabel('X (Size)')
    plt.ylabel('Y (Price)')
    plt.title('Housing Prices - W = {:.2f}, b = {:.2f}'.format(model.w, model.b))
    plt.tight_layout()
    plt.savefig('housing_gd.png', bbox_inches='tight')
    plt.show()

model = MHNeuron()
model.fit(X, y, epochs=100, lr=0.000001)
visualize(X, y, model)
