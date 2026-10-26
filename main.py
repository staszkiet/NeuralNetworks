import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_prime(z):
        return np.ones_like(z)

    @staticmethod
    def sigmoid(z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def sigmoid_prime(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_prime(z):
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_prime(z):
        return np.where(z > 0, 1.0, 0.0)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


class LossFunctions:
    @staticmethod
    def mse(y_hat, y):
        return 0.5 * np.mean(np.sum((y_hat - y) ** 2, axis=0))

    @staticmethod
    def mae(y_hat, y):
        return np.mean(np.sum(np.abs(y_hat - y), axis=0))

    @staticmethod
    def bce(y_hat, y):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat), axis=0))

    @staticmethod
    def cce(y_hat, y):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(np.sum(y * np.log(y_hat), axis=0))


class MultiLayerPerceptron:
    def __init__(self, layer_dims, hidden_activation='sigmoid', seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.L = len(layer_dims) - 1  # Liczba warstw (bez wejściowej)
        self.W = []
        self.b = []
        
        act_map = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_prime),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_prime)
        }
            
        self.hidden_activation, self.hidden_activation_prime = act_map[hidden_activation]
        
        # Inicjalizacja wag i biasów (He initialization)
        for l in range(1, self.L + 1):
            w = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
            b = np.zeros((layer_dims[l], 1))
            self.W.append(w)
            self.b.append(b)

    def forward(self, X):
        A = X
        self.cache = {'A0': A}
        
        for l in range(self.L):
            Z = np.dot(self.W[l], A) + self.b[l]
            self.cache[f'Z{l+1}'] = Z
            
            if l == self.L - 1:
                if self.task_type == 'regression':
                    A = ActivationFunctions.linear(Z)
                elif self.task_type == 'binary_classification':
                    A = ActivationFunctions.sigmoid(Z)
                elif self.task_type == 'multiclass_classification':
                    A = ActivationFunctions.softmax(Z)
            else:
                A = self.hidden_activation(Z)
                
            self.cache[f'A{l+1}'] = A
            
        return A

    def compute_loss(self, Y_hat, Y):
        if self.loss_func == 'mse':
            return LossFunctions.mse(Y_hat, Y)
        elif self.loss_func == 'mae':
            return LossFunctions.mae(Y_hat, Y)
        elif self.loss_func == 'bce':
            return LossFunctions.bce(Y_hat, Y)
        elif self.loss_func == 'cce':
            return LossFunctions.cce(Y_hat, Y)

    def backward(self, Y):
        m = Y.shape[1]
        self.grads = {}
        AL = self.cache[f'A{self.L}']
        Z_last = self.cache[f'Z{self.L}']
        
        # Sygnał błędu delta dla ostatniej warstwy (delta^[L])
        # Zgodnie z wykładem dla sparowanych par (Loss + Activation) wychodzi uproszczenie do (y_hat - y)
        if (self.loss_func == 'mse' and self.task_type == 'regression') or \
           (self.loss_func == 'bce' and self.task_type == 'binary_classification') or \
           (self.loss_func == 'cce' and self.task_type == 'multiclass_classification'):
            dZ = AL - Y
        elif self.loss_func == 'mse' and self.task_type == 'binary_classification':
            # Celowe badanie MSE z sigmoidą (uwaga na nasycenie z wykładu)
            dZ = (AL - Y) * ActivationFunctions.sigmoid_prime(Z_last)
        else:
            # Ogólny przypadek numeryczny
            if self.loss_func == 'mae':
                d_loss = np.sign(AL - Y)
            else:
                d_loss = AL - Y
                
            if self.task_type == 'regression':
                dZ = d_loss * ActivationFunctions.linear_prime(Z_last)
            elif self.task_type == 'binary_classification':
                dZ = d_loss * ActivationFunctions.sigmoid_prime(Z_last)
            else:
                dZ = d_loss

        # Propagacja wsteczna przez warstwy ukryte
        for l in range(self.L, 0, -1):
            A_prev = self.cache[f'A{l-1}']
            self.grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            self.grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                Z_prev = self.cache[f'Z{l-1}']
                dZ = np.dot(self.W[l-1].T, dZ) * self.hidden_activation_prime(Z_prev)

    def update_parameters(self, learning_rate):
        for l in range(1, self.L + 1):
            self.W[l-1] -= learning_rate * self.grads[f'dW{l}']
            self.b[l-1] -= learning_rate * self.grads[f'db{l}']

    def fit(self, X, Y, epochs=100, learning_rate=0.01, batch_size=32, task_type='regression', loss_func='mse'):
        self.task_type = task_type
        self.loss_func = loss_func
        n_samples = X.shape[1]
        history = {'loss': [], 'weights': []}
        
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            
            epoch_loss = 0.0
            n_batches = int(np.ceil(n_samples / batch_size))
            
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]
                
                Y_hat = self.forward(X_batch)
                batch_loss = self.compute_loss(Y_hat, Y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)
                
                self.backward(Y_batch)
                self.update_parameters(learning_rate)
                
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)
            history['weights'].append([w.copy() for w in self.W])
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoka {epoch+1}/{epochs} | Strata ({loss_func.upper()}): {epoch_loss:.6f}")
                
        return history

    def predict(self, X):
        return self.forward(X)