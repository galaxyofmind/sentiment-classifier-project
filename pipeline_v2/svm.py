import pandas as pd 
import numpy as np



class LinearSVM_custom:
    def __init__(self, C=1.0, lr=0.01, epochs=1000, batch_size=1024, decay=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.W = None
        self.b = None
        self.losses = []

    def _dotXw(self, X, W):
        return X.dot(W) if hasattr(X, "dot") else X @ W

    def _to_pm1(self, y):
        y = np.asarray(y).astype(float)
        uniq = set(np.unique(y))
        if uniq == {0.0, 1.0}:
            return np.where(y == 1.0, 1.0, -1.0)
        if uniq == {-1.0, 1.0}:
            return y
        raise ValueError("Labels must be {0,1} or {-1,1}")

    def fit(self, X, y, verbose=False, seed=42):
        rng = np.random.default_rng(seed)
        y_pm1 = self._to_pm1(y)
        m, n = X.shape
        W = np.zeros(n, dtype=np.float32)
        b = 0.0
        m_W = np.zeros(n)
        v_W = np.zeros(n)
        m_b = 0.0
        v_b = 0.0
        losses = []

        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(m)
            lr_t = self.lr / (1.0 + self.decay * epoch)
            for start in range(0, m, self.batch_size):
                sel = idx[start:start + self.batch_size]
                Xb = X[sel]
                yb = y_pm1[sel]
                f = self._dotXw(Xb, W) + b
                margin = yb * f
                viol = margin < 1.0
                if np.any(viol):
                    yv = yb[viol]
                    Xv = Xb[viol]
                    grad_W = W - self.C * (Xv.T @ yv)
                    grad_W /= len(sel)
                    grad_b = float((-self.C * yv.sum()) / len(sel))
                else:
                    grad_W = W / len(sel)
                    grad_b = 0.0
                # Adam update for W
                m_W = self.beta1 * m_W + (1 - self.beta1) * grad_W
                v_W = self.beta2 * v_W + (1 - self.beta2) * (grad_W ** 2)
                m_W_hat = m_W / (1 - self.beta1 ** epoch)
                v_W_hat = v_W / (1 - self.beta2 ** epoch)
                W -= lr_t * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
                # Adam update for b
                m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
                v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)
                m_b_hat = m_b / (1 - self.beta1 ** epoch)
                v_b_hat = v_b / (1 - self.beta2 ** epoch)
                b -= lr_t * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
            if verbose:
                f_all = self._dotXw(X, W) + b
                hinge = np.maximum(0.0, 1.0 - y_pm1 * f_all).mean()
                reg = 0.5 * float(W @ W)
                obj = reg + self.C * hinge * m
                losses.append(hinge)
                if (epoch % 20 == 0 or epoch == self.epochs):
                    print(f"Epoch {epoch:4d} | lr={lr_t:.5f} | obj≈{obj:.6f} (reg={reg:.6f}, hinge≈{hinge:.6f})")
        self.W = W
        self.b = b
        self.losses = losses

    def predict(self, X, return_pm1=False):
        f = self._dotXw(X, self.W) + self.b
        yhat_pm1 = np.where(f >= 0.0, 1.0, -1.0)
        if return_pm1:
            return yhat_pm1
        return (yhat_pm1 == 1.0).astype(int)

    def confusion(self, X, y_true):
        y_pred = self.predict(X)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[tn, fp], [fn, tp]])
    