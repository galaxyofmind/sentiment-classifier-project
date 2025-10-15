# Data split strategies

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np

class DataSplitter:
    def __init__(self, df: pd.DataFrame, text_col: str = "content", label_col: str = "label"):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.X = df[text_col].values
        self.y = df[label_col].values
        self.holdout_X_train, self.holdout_y_train, self.holdout_X_test, self.holdout_y_test = self.hold_out_split(test_size=0.25, random_state=42)
        self.strat_X_train, self.strat_X_test, self.strat_y_train, self.strat_y_test = self.stratified_split(test_size=0.25, random_state=42)
        self.kfolds = self.k_fold_split(k=5)
        self.strat_kfold = self.stratified_kfold_split()

    def hold_out_split(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu theo chiến lược Hold-out
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return X_train, y_train, X_test, y_test

    def k_fold_split(self, k=5, shuffle=False, random_state=None):
        """
        Trả về danh sách các fold: mỗi fold là tuple (X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        
        return kf

    def stratified_split(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu theo chiến lược Stratified Sampling
        Đảm bảo tỷ lệ nhãn cân bằng giữa train/test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        return X_train, X_test, y_train, y_test
    
    def stratified_kfold_split(self, k=5, shuffle=True, random_state=42):
        """
        Chia dữ liệu theo Stratified K-Fold Cross-validation
        Đảm bảo tỷ lệ nhãn giống nhau trong mỗi fold
        """
        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        splits = []
        for train_idx, test_idx in skf.split(self.X, self.y):
            splits.append(((self.X[train_idx], self.y[train_idx]),
                        (self.X[test_idx], self.y[test_idx])))
        return splits




