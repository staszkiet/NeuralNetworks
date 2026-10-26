import numpy as np
import pandas as pd
import os

from main import MultiLayerPerceptron

class MLPExperimentRunner:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.model = None
        self.task_type = None
        self.loss_func = None
        
    def load_dataset(self, filename):
        """
        Wczytuje plik CSV:
        - Klasyfikacja (x, y, cls): zwraca X o kształcie (2, N) oraz Y
        - Regresja (x, y): zwraca X o kształcie (1, N) oraz Y
        """
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath)
        
        if 'cls' in df.columns:
            self.task_type = 'classification'
            X = df[['x', 'y']].values.T  # Kształt: (2, n_samples)
            y_raw = df['cls'].values
            
            classes = np.unique(y_raw)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([class_to_idx[c] for c in y_raw])
            if len(classes) == 2:
                # Klasyfikacja binarna (0/1 pod sigmoid + BCE)
                Y = y_idx.reshape(1, -1).astype(float)
            else:
                # Klasyfikacja wieloklasowa (One-Hot)
                Y = np.zeros((len(classes), len(y_raw)))
                for idx in range(len(classes)):
                    Y[idx, y_idx == idx] = 1.0
        else:
            self.task_type = 'regression'
            X = df[['x']].values.T      # Kształt: (1, n_samples)
            Y = df[['y']].values.T      # Kształt: (1, n_samples)
            
        return X, Y

    def configure_model(self, layer_dims, hidden_activation='sigmoid', loss_func='mse', seed=42):
        """
        Konfiguracja modelu w jednym miejscu:
        - layer_dims: lista określająca liczbę neuronów w warstwach
        - hidden_activation: 'sigmoid', 'tanh' lub 'relu'
        - loss_func: 'mse', 'mae', 'bce' lub 'cce'
        - seed: ziarno generatora losowego
        """
        self.loss_func = loss_func
        self.model = MultiLayerPerceptron(
            layer_dims=layer_dims, 
            hidden_activation=hidden_activation, 
            seed=seed
        )

    def run_experiment(self, train_file, epochs=500, learning_rate=0.01, batch_size=32):
        """
        Uruchamia trening na podstawie wcześniej skonfigurowanego modelu.
        """
            
        X_train, Y_train = self.load_dataset(train_file)
        
        # Wyznaczenie szczegółowego typu zadania na podstawie wczytanych danych
        if self.task_type == 'classification':
            task_type_str = 'binary_classification' if Y_train.shape[0] == 1 else 'multiclass_classification'
        else:
            task_type_str = 'regression'
            
        print(f"Uruchomienie eksperymentu | Zadanie: {task_type_str} | Loss: {self.loss_func.upper()} | Aktywacja: {self.model.hidden_activation.__name__}")
        
        history = self.model.fit(
            X_train, Y_train, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            task_type=task_type_str,
            loss_func=self.loss_func,
        )
        
        return history

    def evaluate(self, test_file):
        """
        Ewaluacja wyuczonego modelu na zbiorze testowym.
        """
        X_test, Y_test = self.load_dataset(test_file)
        Y_pred = self.model.predict(X_test)
        
        loss = self.model.compute_loss(Y_pred, Y_test)
        
        if self.task_type == 'classification':
            if Y_test.shape[0] == 1:
                predictions = (Y_pred >= 0.5).astype(int)
                accuracy = np.mean(predictions == Y_test)
            else:
                predictions = np.argmax(Y_pred, axis=0)
                true_classes = np.argmax(Y_test, axis=0)
                accuracy = np.mean(predictions == true_classes)
            print(f"Test -> Strata ({self.loss_func.upper()}): {loss:.4f} | Dokładność (Accuracy): {accuracy * 100:.2f}%")
            return loss, accuracy
        else:
            print(f"Test -> Strata ({self.loss_func.upper()}): {loss:.4f}")
            return loss