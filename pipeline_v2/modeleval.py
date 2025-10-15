import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from vectorization import Vectorizer_TFIDF

from naive_bayes import NaiveBayes
from svm import LinearSVM_custom
from logistic_regression import LogisticRegression_custom


class ModelEvaluator:
    """
    Class để huấn luyện và đánh giá nhiều mô hình ML
    Tương thích với DataSplitter và Vectorizer_TFIDF
    """
    
    def __init__(self, models=None, param=None):
        """
        Parameters:
        -----------
        models : dict, optional
            Dictionary chứa các model {tên: model_instance}
            Nếu None, sẽ dùng bộ models mặc định
        """
        if models is None:
            self.models = {
                "SVM": LinearSVC(random_state=42),
                "Naive Bayes": MultinomialNB(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            }
        elif models == "custom":
            self.models = {
                "SVM": LinearSVM_custom(),
                "Naive Bayes": NaiveBayes(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression_custom()
            }
        elif models == "optimize":
            self.models = {
                "SVM": LinearSVC(C=1, max_iter=40, random_state=42),
                "Naive Bayes": MultinomialNB(alpha=0.1),
                "Random Forest": RandomForestClassifier(n_estimators=40, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=35, random_state=42)
            }
        
        
        self.results = []
        self.trained_models = {}
        self.vectorizer = None
    
    def evaluate(self, X_train, y_train, X_test, y_test, show_cm=True, cm_figsize=(10, 8), show_metric=True):
        """
        Huấn luyện và đánh giá các models với dữ liệu đã vectorize
        
        Parameters:
        -----------
        X_train : array-like or sparse matrix
            Dữ liệu training đã vectorize
        y_train : array-like
            Nhãn training
        X_test : array-like or sparse matrix
            Dữ liệu test đã vectorize
        y_test : array-like
            Nhãn test
        show_cm : bool, default=True
            Hiển thị confusion matrix
        cm_figsize : tuple, default=(6, 6)
            Kích thước figure cho confusion matrix
        
        Returns:
        --------
        pd.DataFrame
            Bảng kết quả đánh giá các model
        """
        self.results = []
        cms = {}
        
        for name, model in self.models.items():
            #print(f"Training {name}...")
            
            # Huấn luyện
            model.fit(X_train, y_train)
            
            # Dự đoán
            y_pred = model.predict(X_test)
            
            # Tính các metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            self.results.append([name, acc, prec, rec, f1])
            
            # Lưu model đã train
            self.trained_models[name] = model
            
            # Lưu confusion matrix
            if show_cm:
                cms[name] = confusion_matrix(y_test, y_pred)


        if show_cm:
            self.__plot_cms__(cms=cms,cm_figsize=cm_figsize)

        
        # Tạo DataFrame kết quả
        df_results = pd.DataFrame(
            self.results, 
            columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
        )

        if show_metric:
            self.__plot_results_table__(df_results)

        return df_results
    
    def evaluate_with_vectorizer(self, data_splitter, vectorizer, 
                                  split_type="stratified", 
                                  show_cm=True, cm_figsize=(10, 8),
                                  show_vectorizer_info=True,
                                  show_metric=True):
        """
        Đánh giá model với DataSplitter và Vectorizer_TFIDF
        
        Parameters:
        -----------
        data_splitter : DataSplitter
            Object DataSplitter đã được khởi tạo
        vectorizer : Vectorizer_TFIDF
            Object Vectorizer_TFIDF (chưa cần fit trước)
        split_type : str, default="stratified"
            Loại split: "holdout", "stratified", "kfold", "stratified_kfold"
        show_cm : bool, default=True
            Hiển thị confusion matrix
        cm_figsize : tuple, default=(6, 6)
            Kích thước figure cho confusion matrix
        show_vectorizer_info : bool, default=True
            Hiển thị thông tin vectorizer
        
        Returns:
        --------
        pd.DataFrame hoặc dict
            Kết quả đánh giá
        """
        self.vectorizer = vectorizer
        
        if split_type == "holdout":
            print("=== Using Holdout Split ===")
            X_train = vectorizer.fit_transform(data_splitter.holdout_X_train)
            X_test = vectorizer.transform(data_splitter.holdout_X_test)
            
            if show_vectorizer_info:
                vectorizer.info(X_train)
            
            return self.evaluate(
                X_train,
                data_splitter.holdout_y_train,
                X_test,
                data_splitter.holdout_y_test,
                show_cm=show_cm,
                cm_figsize=cm_figsize,
            )
        
        elif split_type == "stratified":
            print("=== Using Stratified Split ===")
            X_train = vectorizer.fit_transform(data_splitter.strat_X_train)
            X_test = vectorizer.transform(data_splitter.strat_X_test)
            
            if show_vectorizer_info:
                vectorizer.info(X_train)
            
            return self.evaluate(
                X_train,
                data_splitter.strat_y_train,
                X_test,
                data_splitter.strat_y_test,
                show_cm=show_cm,
                cm_figsize=cm_figsize,
                show_metric=show_metric
            )
        
        elif split_type == "kfold":
            print("=== Using K-Fold Cross Validation ===")
            return self._evaluate_kfold(
                data_splitter.kfolds,
                data_splitter.X,
                data_splitter.y,
                vectorizer,
                show_cm=show_cm,
                show_vectorizer_info=show_vectorizer_info
            )
        
        elif split_type == "stratified_kfold":
            print("=== Using Stratified K-Fold Cross Validation ===")
            return self._evaluate_stratified_kfold(
                data_splitter.strat_kfold,
                vectorizer,
                show_cm=show_cm,
                show_vectorizer_info=show_vectorizer_info
            )
        
        else:
            raise ValueError(f"Unknown split_type: {split_type}. Choose from: 'holdout', 'stratified', 'kfold', 'stratified_kfold'")
    
    def _evaluate_kfold(self, kfolds, X, y, vectorizer, show_cm=False, show_vectorizer_info=False):
        """Đánh giá với K-Fold"""
        fold_results = {name: [] for name in self.models.keys()}
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            #print(f"\n--- Fold {fold_idx + 1} ---")
            
            # Vectorize dữ liệu
            X_train_vec = vectorizer.fit_transform(X_train_raw)
            X_test_vec = vectorizer.transform(X_test_raw)
            
            if show_vectorizer_info and fold_idx == 0:
                vectorizer.info(X_train_vec)
            
            for name, model in self.models.items():
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                
                fold_results[name].append({
                    "fold": fold_idx + 1,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                })
                
                #print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # Tính trung bình và std các fold
        avg_results = []
        for name, scores in fold_results.items():
            avg_acc = np.mean([s["accuracy"] for s in scores])
            std_acc = np.std([s["accuracy"] for s in scores])
            avg_prec = np.mean([s["precision"] for s in scores])
            std_prec = np.std([s["precision"] for s in scores])
            avg_rec = np.mean([s["recall"] for s in scores])
            std_rec = np.std([s["recall"] for s in scores])
            avg_f1 = np.mean([s["f1"] for s in scores])
            std_f1 = np.std([s["f1"] for s in scores])
            
            avg_results.append([
                name, 
                f"{avg_acc:.4f} ± {std_acc:.4f}",
                f"{avg_prec:.4f} ± {std_prec:.4f}",
                f"{avg_rec:.4f} ± {std_rec:.4f}",
                f"{avg_f1:.4f} ± {std_f1:.4f}"
            ])
        
        df_avg = pd.DataFrame(
            avg_results,
            columns=["Model", "Accuracy (mean±std)", "Precision (mean±std)", 
                    "Recall (mean±std)", "F1-score (mean±std)"]
        )

        self.__plot_results_table__("summary: \n", df_avg)
        
        return df_avg
    
    def _evaluate_stratified_kfold(self, strat_kfold_splits, vectorizer, 
                                    show_cm=False, show_vectorizer_info=False):
        """Đánh giá với Stratified K-Fold"""
        fold_results = {name: [] for name in self.models.keys()}
        
        for fold_idx, ((X_train_raw, y_train), (X_test_raw, y_test)) in enumerate(strat_kfold_splits):
            #print(f"\n--- Stratified Fold {fold_idx + 1} ---")
            
            # Vectorize dữ liệu
            X_train_vec = vectorizer.fit_transform(X_train_raw)
            X_test_vec = vectorizer.transform(X_test_raw)
            
            if show_vectorizer_info and fold_idx == 0:
                vectorizer.info(X_train_vec)
            
            for name, model in self.models.items():
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                
                fold_results[name].append({
                    "fold": fold_idx + 1,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                })
                
                #print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # Tính trung bình và std các fold
        avg_results = []
        for name, scores in fold_results.items():
            avg_acc = np.mean([s["accuracy"] for s in scores])
            std_acc = np.std([s["accuracy"] for s in scores])
            avg_prec = np.mean([s["precision"] for s in scores])
            std_prec = np.std([s["precision"] for s in scores])
            avg_rec = np.mean([s["recall"] for s in scores])
            std_rec = np.std([s["recall"] for s in scores])
            avg_f1 = np.mean([s["f1"] for s in scores])
            std_f1 = np.std([s["f1"] for s in scores])
            
            avg_results.append([
                name,
                f"{avg_acc:.4f} ± {std_acc:.4f}",
                f"{avg_prec:.4f} ± {std_prec:.4f}",
                f"{avg_rec:.4f} ± {std_rec:.4f}",
                f"{avg_f1:.4f} ± {std_f1:.4f}"
            ])
        
        df_avg = pd.DataFrame(
            avg_results,
            columns=["Model", "Accuracy (mean±std)", "Precision (mean±std)", 
                    "Recall (mean±std)", "F1-score (mean±std)"]
        )

        print("summary: \n")
        self.__plot_results_table__(df_avg)
        
        return df_avg
    
    # def get_best_model(self, metric="F1-score"):
    #     """
    #     Lấy model tốt nhất theo metric
        
    #     Parameters:
    #     -----------
    #     metric : str, default="F1-score"
    #         Metric để chọn model tốt nhất: "Accuracy", "Precision", "Recall", "F1-score"
        
    #     Returns:
    #     --------
    #     tuple
    #         (tên_model, model_instance, score)
    #     """
    #     if not self.results:
    #         raise ValueError("Chưa có kết quả đánh giá. Chạy evaluate() hoặc evaluate_with_vectorizer() trước.")
        
    #     df = pd.DataFrame(self.results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
    #     best_idx = df[metric].idxmax()
    #     best_name = df.loc[best_idx, "Model"]
    #     best_score = df.loc[best_idx, metric]
        
    #     return best_name, self.trained_models[best_name], best_score
    
    def predict(self, X_raw, model_name=None):
        """
        Dự đoán với dữ liệu mới
        
        Parameters:
        -----------
        X_raw : array-like
            Dữ liệu text chưa vectorize
        model_name : str, optional
            Tên model để dự đoán. Nếu None, dùng model tốt nhất
        
        Returns:
        --------
        array
            Predictions
        """
        if self.vectorizer is None:
            raise ValueError("Chưa có vectorizer. Chạy evaluate_with_vectorizer() trước.")
        
        if model_name is None:
            model_name, model, _ = self.get_best_model()
            print(f"Sử dụng model tốt nhất: {model_name}")
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' chưa được train.")
            model = self.trained_models[model_name]
        
        # Vectorize và dự đoán
        X_vec = self.vectorizer.transform(X_raw)
        predictions = model.predict(X_vec)
        
        return predictions
    
    def __plot_results_table__(self, df_results, figsize=(8,4), cmap="Blues"):
        # extract only mean values from "mean ± std" strings
        metric_cols = [c for c in df_results.columns if c != "Model"]

        df_numeric = df_results.copy()
        for col in metric_cols:
            df_numeric[col] = df_numeric[col].apply(
                lambda x: float(str(x).split("±")[0].strip()) if isinstance(x, str) and "±" in x else float(x)
            )

        plt.figure(figsize=figsize)
        metrics = df_numeric.drop(columns=["Model"])

        sns.heatmap(
            metrics,
            annot=True, fmt=".3f", cmap=cmap, cbar=False,
            xticklabels=metrics.columns,
            yticklabels=df_results["Model"]
        )

        plt.title("Model Evaluation Results")
        plt.ylabel("Model")
        plt.xlabel("Metrics")
        plt.tight_layout()
        plt.show()

    def __plot_cms__(self, cms, cm_figsize=(10,8)):
        if cms:
            fig, axes = plt.subplots(2, 2, figsize=cm_figsize)
            axes = axes.ravel()

            for i, (name, cm) in enumerate(cms.items()):
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
                axes[i].set_title(f"Confusion Matrix - {name}")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("True")

            # Ẩn subplot trống nếu < 4 models
            for j in range(i + 1, 4):
                fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def evaluate_with_min_df(self, data, min_df_values=range(1, 20, 1), max_features=30000, split_type="stratified"):
        results = {m: [] for m in ["SVM", "Naive Bayes", "Random Forest", "Logistic Regression"]}

        for i in min_df_values:
            print(f"=== Evaluating with min_df={i} ===")

            vectorizer = Vectorizer_TFIDF(
                ngram_range=(1,2), 
                min_df=i, 
                max_features=max_features
            )

            # chạy evaluate
            df_avg = self.evaluate_with_vectorizer(
                data, 
                vectorizer,
                split_type=split_type,
                show_cm=False,
                show_vectorizer_info=True,
                show_metric=False
            )

            # lấy kết quả cho từng model (không phải dòng cuối cùng)
            for model in results.keys():
                row = df_avg[df_avg["Model"] == model].iloc[0]
                results[model].append([i, row["Accuracy"], row["Precision"], row["Recall"], row["F1-score"]])

        # Vẽ biểu đồ cho từng model
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (model, data_points) in enumerate(results.items()):
            df_model = pd.DataFrame(data_points, columns=["min_df", "Accuracy", "Precision", "Recall", "F1-score"])
            
            for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
                means = df_model[metric].astype(str).str.split("±").str[0].astype(float)
                axes[idx].plot(df_model["min_df"], means, marker="o", label=metric)
            
            axes[idx].set_title(model)
            axes[idx].set_xlabel("min_df")
            axes[idx].set_ylabel("Score")
            axes[idx].grid(True)
            axes[idx].legend()

        plt.suptitle("Impact of min_df on Model Performance", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return results
    
    def track_rf_trees_performance(self, X_train, y_train, X_test, y_test,
                                tree_list=[10, 50, 100, 200, 500]):
        """
        Trả về DataFrame chứa Accuracy, Precision, Recall, F1 khi thay đổi n_estimators
        (các tham số khác để mặc định).
        """
        results = []

        for n_trees in tree_list:
            model = RandomForestClassifier(
                n_estimators=n_trees,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append([n_trees, acc, prec, rec, f1])

        df_results = pd.DataFrame(results, columns=["n_estimators", "Accuracy", "Precision", "Recall", "F1-score"])

        # Vẽ trực quan hóa
        plt.figure(figsize=(10, 6))
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            plt.plot(df_results["n_estimators"], df_results[metric], marker="o", label=metric)

        plt.xlabel("n_estimators (số cây)")
        plt.ylabel("Score")
        plt.title("Ảnh hưởng của số cây đến Random Forest")
        plt.legend()
        plt.grid(True)
        plt.show()

        return df_results

    
    def track_C_performance(self, X_train, y_train, X_test, y_test, C_values, model_type="svm"):
        """
        model_type: 'logreg' hoặc 'svm'
        X_train, y_train, X_test, y_test: dữ liệu
        C_values: list giá trị C muốn thử
        """
        results = []

        for C in C_values:
            if model_type == "logreg":
                model = LogisticRegression(C=C, max_iter=1000, random_state=42)
            elif model_type == "svm":
                model = LinearSVC(C=C, random_state=42)
            else:
                raise ValueError("Chỉ hỗ trợ 'logreg' hoặc 'svm'")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append([C, acc, prec, rec, f1])

        # Kết quả DataFrame
        df_results = pd.DataFrame(results, columns=["C", "Accuracy", "Precision", "Recall", "F1-score"])

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["C"], df_results["Accuracy"], label="Accuracy", marker="o")
        plt.plot(df_results["C"], df_results["Precision"], label="Precision", marker="o")
        plt.plot(df_results["C"], df_results["Recall"], label="Recall", marker="o")
        plt.plot(df_results["C"], df_results["F1-score"], label="F1-score", marker="o")

        plt.xscale("log")  # log scale giúp dễ nhìn khi C thay đổi nhiều
        plt.xlabel("C (log scale)")
        plt.ylabel("Score")
        plt.title(f"Ảnh hưởng của tham số C đến {model_type.upper()}")
        plt.legend()
        plt.grid(True)
        plt.show()

        return df_results

    def track_rf_hyperparams(self, X_train, y_train, X_test, y_test,
                         n_estimators_list=[10, 50, 100],
                         max_depth_list=[None, 5, 10],
                         min_samples_split_list=[2, 5, 10],
                         min_samples_leaf_list=[1, 2, 5]):
        """
        Theo dõi Accuracy, Precision, Recall, F1-score khi thay đổi các hyperparameter của Random Forest.
        """

        results = []

        for n in n_estimators_list:
            for d in max_depth_list:
                for s in min_samples_split_list:
                    for l in min_samples_leaf_list:
                        model = RandomForestClassifier(
                            n_estimators=n,
                            max_depth=d,
                            min_samples_split=s,
                            min_samples_leaf=l,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                        results.append([n, d, s, l, acc, prec, rec, f1])

        # Tạo DataFrame kết quả
        df_results = pd.DataFrame(
            results,
            columns=["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                    "Accuracy", "Precision", "Recall", "F1-score"]
        )

        # --- Visualization ---
        # Vẽ biểu đồ ảnh hưởng của từng tham số riêng lẻ
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            axes[0,0].plot(df_results["n_estimators"], df_results[metric], marker="o", label=metric)
        axes[0,0].set_title("Impact of n_estimators")
        axes[0,0].set_xlabel("n_estimators")
        axes[0,0].set_ylabel("Score")
        axes[0,0].legend()
        axes[0,0].grid(True)

        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            axes[0,1].plot(df_results["max_depth"].astype(str), df_results[metric], marker="s", label=metric)
        axes[0,1].set_title("Impact of max_depth")
        axes[0,1].set_xlabel("max_depth")
        axes[0,1].set_ylabel("Score")
        axes[0,1].legend()
        axes[0,1].grid(True)

        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            axes[1,0].plot(df_results["min_samples_split"], df_results[metric], marker="^", label=metric)
        axes[1,0].set_title("Impact of min_samples_split")
        axes[1,0].set_xlabel("min_samples_split")
        axes[1,0].set_ylabel("Score")
        axes[1,0].legend()
        axes[1,0].grid(True)

        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            axes[1,1].plot(df_results["min_samples_leaf"], df_results[metric], marker="d", label=metric)
        axes[1,1].set_title("Impact of min_samples_leaf")
        axes[1,1].set_xlabel("min_samples_leaf")
        axes[1,1].set_ylabel("Score")
        axes[1,1].legend()
        axes[1,1].grid(True)

        plt.tight_layout()
        plt.show()

        return df_results


    def track_logreg_performance(self, X_train, y_train, X_test, y_test,
                             class_weight="balanced",
                             max_iters=[100, 200, 500, 1000]):
        """
        Trả về DataFrame chứa Accuracy, Precision, Recall, F1 khi thay đổi max_iter
        (giữ nguyên class_weight).
        """
        results = []

        for mi in max_iters:
            model = LogisticRegression(
                class_weight=class_weight,
                max_iter=mi,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append([mi, acc, prec, rec, f1])

        df_results = pd.DataFrame(results, columns=["max_iter", "Accuracy", "Precision", "Recall", "F1-score"])

        # Vẽ trực quan hóa
        plt.figure(figsize=(10, 6))
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            plt.plot(df_results["max_iter"], df_results[metric], marker="o", label=metric)

        plt.xlabel("max_iter")
        plt.ylabel("Score")
        plt.title(f"Ảnh hưởng của max_iter đến Logistic Regression (class_weight={class_weight})")
        plt.legend()
        plt.grid(True)
        plt.show()

        return df_results

    
    def track_nb_alpha(self, X_train, y_train, X_test, y_test, alpha_list=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """
        Theo dõi Accuracy, Precision, Recall, F1-score khi thay đổi hyperparameter alpha của Naive Bayes.
        """

        results = []

        for a in alpha_list:
            model = MultinomialNB(alpha=a)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append([a, acc, prec, rec, f1])

        # Tạo DataFrame kết quả
        df_results = pd.DataFrame(
            results,
            columns=["alpha", "Accuracy", "Precision", "Recall", "F1-score"]
        )

        # --- Visualization ---
        plt.figure(figsize=(10, 6))
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
            plt.plot(df_results["alpha"], df_results[metric], marker="o", label=metric)
        plt.title("Impact of alpha on Naive Bayes Performance")
        plt.xlabel("alpha")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

        return df_results

