import numpy as np
import pandas as pd

# Data training untuk bioinformatika
X_train = np.array([[3, 4], [1, 4], [2, 3], [6, 8], [7, 7], [8, 5]])
y_train = np.array([1, 1, 1, -1, -1, -1])

# Data testing untuk bioinformatika
X_test = np.array([[5, 6], [2, 1], [7,10]])

class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inisialisasi weights dan bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent untuk optimasi SVM
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * 1 / self.epochs * self.weights)
                else:
                    self.weights -= self.lr * (2 * 1 / self.epochs * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

# Inisialisasi dan training model SVM
model = SVM()
model.fit(X_train, y_train)

# Prediksi kelas untuk data testing
predictions = model.predict(X_test)

# Menampilkan data training dalam bentuk tabel
df_train = pd.DataFrame(np.column_stack([X_train[:, 0], X_train[:, 1], y_train]), columns=['Feature 1', 'Feature 2', 'Label'])
print("Data Training:")
print(df_train)

# Menampilkan data testing dalam bentuk tabel
df_test = pd.DataFrame(X_test[:, 0], columns=['Feature 1'])
df_test['Feature 2'] = X_test[:, 1]  # Tambah kolom Feature 2
print("\nData Testing:")
print(df_test)

# Menampilkan hasil prediksi
print("\nPrediksi untuk data testing:")
for i, pred in enumerate(predictions):
    print(f"Data {i+1}: Prediksi kelas = {int(pred)}")
