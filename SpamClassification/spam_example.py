import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spam_visualization import readSpamDataset


# Define the inputs and outputs
df = readSpamDataset()
X = df[['spam_cnt', 'ham_cnt']].values
y = df['label'].values.astype(np.int8)[np.newaxis, :].T

class MHNeuron:
    def __init__(self, w=None, b=None):
        self.w = w if w is not None else 10 * np.random.randn(2, 1)
        self.b = b if b is not None else np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self.sigmoid(x.dot(self.w) + self.b)
    
    def fit(self, X, y, epochs=10, lr=0.1):
        m = len(X)
        J_list = [self.J(X, y)]
        acc_list = [accuracy(X, y, self)]
        for epoch in range(epochs):
            y_hat = self.forward(X)
            error = y - y_hat
            self.w += lr / m * X.T.dot(error)
            self.b += lr / m * np.sum(error)
            J_list.append(self.J(X, y))
            acc_list.append(accuracy(X, y, self))

        return J_list, acc_list

    def J(self, X, y):
        return -np.mean((1 - y) * np.log(1 - self.forward(X)) + y * np.log(self.forward(X)))
    
def accuracy(X, y, model):
    y_hat = model.forward(X)
    y_hat[y_hat < 0.5] = 0
    y_hat[y_hat >= 0.5] = 1
    return np.mean(y_hat == y)

def visualize(X, y, model):
    X_min, X_max = X.min() - 0.5, X.max() + 0.5
    y_hat = model.forward(np.linspace(X_min, X_max, 100)[np.newaxis, :].T)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='red', s=70)
    plt.plot(np.linspace(X_min, X_max, 100), y_hat, color='green', linewidth=3)
    plt.xlabel('X (Size)')
    plt.ylabel('Y (Price)')
    plt.title('Spam Classification - W = {:.2f}, b = {:.2f}'.format(model.w, model.b))
    plt.tight_layout()
    plt.savefig('spam_gd.png', bbox_inches='tight')
    plt.show()

model = MHNeuron()
print(f'Initial J = {model.J(X, y):4.2f}, Initial Accuracy = {accuracy(X, y, model):4.2f}')
J_list, acc_list = model.fit(X, y, epochs=100000, lr=0.1)
print(f'Final J = {model.J(X, y):4.2f}', f'Final Accuracy = {accuracy(X, y, model):4.2f}')

##  Plot J and accuracy...
sns.set(font_scale=1.5)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(J_list, linewidth=3)
plt.xlabel('Epoch')
plt.ylabel('J')
plt.title('Cost Function')
plt.subplot(1, 2, 2)
plt.plot(acc_list, linewidth=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.tight_layout()
plt.savefig('spam_gd_cost_acc.png', bbox_inches='tight')
plt.show()