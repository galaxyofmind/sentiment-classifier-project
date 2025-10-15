import joblib
import pandas as pd 
import numpy as np


# 4.1 Mô hình  Logistic Regression

class LogisticRegression_custom:
    def __init__(self, lr=0.01, epochs=1000, l2=1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.W = None
        self.b = None
        self.losses = []

    def _sigmoid(self, x):
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)
        return out

    def fit(self, X, y, beta1=0.9, beta2=0.999, eps=1e-8, verbose=False):
        y = np.asarray(y).astype(np.float32)
        #if set(np.unique(y)) == {-1.0, 1.0}:
        #    y = (y == 1.0).astype(np.float32)
        m, n = X.shape
        W = np.zeros(n, dtype=np.float32)
        b = 0.0
        m_W = np.zeros(n)
        v_W = np.zeros(n)
        m_b = 0.0
        v_b = 0.0
        losses = []
        for epoch in range(1, self.epochs + 1):
            z = X @ W + b
            p = self._sigmoid(z)
            err = (p - y).astype(np.float32)
            grad_W = X.T @ err / m + self.l2 * W
            grad_b = float(err.mean())
            # Adam update for W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
            m_W_hat = m_W / (1 - beta1 ** epoch)
            v_W_hat = v_W / (1 - beta2 ** epoch)
            W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            # Adam update for b
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            m_b_hat = m_b / (1 - beta1 ** epoch)
            v_b_hat = v_b / (1 - beta2 ** epoch)
            b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)) + self.l2 * np.sum(W ** 2) / 2
            losses.append(loss)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        self.W = W
        self.b = b
        self.losses = losses

    def predict(self, X, return_proba=False, threshold=0.5):
        z = X @ self.W + self.b
        prob = self._sigmoid(z)
        if return_proba:
            return prob
        return (prob >= threshold).astype(int)

    def confusion(self, X, y_true, threshold=0.5):
        y_pred = self.predict(X, threshold=threshold)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[tn, fp], [fn, tp]])
    
# 5.1 Logistic regression model
# lr_model = LogisticRegression_custom(lr=0.1, epochs=300, l2=1e-4)

# params = np.load("model/logistic_weights.npz")
# lr_model.W = params["W"]
# lr_model.b = params["b"]