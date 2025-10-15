import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self, model="multinomial"):
        """
        model: 'multinomial' for counts (text data),
        'gaussian' for continuous features
        """
        self.model = model
        self.class_priors = {}
        self.feature_probs = {}
        self.class_stats = {}

    def fit(self, X, y):
        y = np.array(y)
        self.classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        # Prior probabilities
        self.class_priors = {c: counts[i]/total_samples for i, c in enumerate(self.classes)}

        if self.model == "multinomial":
            # Multinomial NB (for discrete counts, e.g., text word frequencies)
            n_classes = len(self.classes)
            n_features = X.shape[1]
            self.feature_probs = {c: np.zeros(n_features) for c in self.classes}

            for c in self.classes:
                X_c = X[y == c]
                feature_counts = X_c.sum(axis=0)
                total_count = feature_counts.sum()
                # Laplace smoothing
                self.feature_probs[c] = (feature_counts + 1) / (total_count + n_features)

        elif self.model == "gaussian":
            # Gaussian NB (for continuous features)
            self.class_stats = {}
            for c in self.classes:
                X_c = X[y == c]
                mean = X_c.mean(axis=0)
                var = X_c.var(axis=0) + 1e-9  # avoid divide by zero
                self.class_stats[c] = (mean, var)

    def _multinomial_log_prob(self, x, c):
        log_prior = np.log(self.class_priors[c])
        log_likelihood = np.sum(x * np.log(self.feature_probs[c]))
        return log_prior + log_likelihood

    def _gaussian_log_prob(self, x, c):
        log_prior = np.log(self.class_priors[c])
        mean, var = self.class_stats[c]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
        return log_prior + log_likelihood

    def predict(self, X):
        preds = []
        for x in X:
            if self.model == "multinomial":
                log_probs = {c: self._multinomial_log_prob(x, c) for c in self.classes}
            else:
                log_probs = {c: self._gaussian_log_prob(x, c) for c in self.classes}
            preds.append(max(log_probs, key=log_probs.get))
        return np.array(preds)

    def predict_proba(self, X):
        probs = []
        for x in X:
            if self.model == "multinomial":
                log_probs = {c: self._multinomial_log_prob(x, c) for c in self.classes}
            else:
                log_probs = {c: self._gaussian_log_prob(x, c) for c in self.classes}
            max_log = max(log_probs.values())
            exp_probs = {c: np.exp(v - max_log) for c, v in log_probs.items()}  # stabilize
            total = sum(exp_probs.values())
            probs.append([exp_probs[c]/total for c in self.classes])
        return np.array(probs)
