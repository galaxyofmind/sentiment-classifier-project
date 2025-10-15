import numpy as np
from collections import Counter
import math
import random
from sklearn.metrics import accuracy_score


class ID3Tree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def information_gain(self, X_col, y, threshold):
        left_idx = X_col <= threshold
        right_idx = X_col > threshold
        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return 0
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = sum(left_idx), sum(right_idx)
        e_left, e_right = self.entropy(y[left_idx]), self.entropy(y[right_idx])
        child_entropy = (n_left/n)*e_left + (n_right/n)*e_right
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        n_features = X.shape[1]
        for feat in range(n_features):
            values = np.unique(X[:, feat])
            for threshold in values:
                gain = self.information_gain(X[:, feat], y, threshold)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, threshold
        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]
        if self.max_depth and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]
        if len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]
        feat, thresh = self.best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0]
        left_idx = X[:, feat] <= thresh
        right_idx = X[:, feat] > thresh
        left_subtree = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_subtree = self.build_tree(X[right_idx], y[right_idx], depth+1)
        return (feat, thresh, left_subtree, right_subtree)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        if x[feat] <= thresh:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class RandomForestID3:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2,
                 sample_size=None, max_features=None, oob_score=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.max_features = max_features
        self.oob_score = oob_score
        self.trees = []
        self.oob_indices = []
        self.oob_score_ = None

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(
            n_samples,
            n_samples if self.sample_size is None else self.sample_size,
            replace=True
        )
        oob = np.setdiff1d(np.arange(n_samples), idxs)
        return X[idxs], y[idxs], oob

    def fit(self, X, y):

        self.trees = []
        self.oob_indices = []
        for _ in range(self.n_trees):
            X_samp, y_samp, oob = self.bootstrap_sample(X, y)
            self.oob_indices.append(oob)

            # chọn ngẫu nhiên tập con feature
            if self.max_features:
                feat_idxs = np.random.choice(X.shape[1], self.max_features, replace=False)
            else:
                feat_idxs = np.arange(X.shape[1])

            tree = ID3Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_samp[:, feat_idxs], y_samp)
            self.trees.append((tree, feat_idxs))

        if self.oob_score:
            self._compute_oob_score(X, y)


    def _compute_oob_score(self, X, y):
        oob_votes = {i: [] for i in range(len(y))}
        for (tree, feat_idxs), oob in zip(self.trees, self.oob_indices):
            if len(oob) == 0:
                continue
            preds = tree.predict(X[oob][:, feat_idxs])
            for idx, p in zip(oob, preds):
                oob_votes[idx].append(p)
        final_preds, true_labels = [], []
        for idx, votes in oob_votes.items():
            if len(votes) > 0:
                pred = max(set(votes), key=votes.count)
                final_preds.append(pred)
                true_labels.append(y[idx])
        if len(final_preds) > 0:
            self.oob_score_ = accuracy_score(true_labels, final_preds)
        else:
            self.oob_score_ = None

    def predict(self, X):

        tree_preds = np.array([
            tree.predict(X[:, feat_idxs]) for tree, feat_idxs in self.trees
        ])
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # (n_samples, n_trees)
        y_pred = [Counter(row).most_common(1)[0][0] for row in tree_preds]
        return np.array(y_pred)
    
    