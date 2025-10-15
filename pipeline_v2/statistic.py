import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import re
from preprocessNLP import Preprocessor
import underthesea

class SentimentStats:
    def __init__(self, df, text_col="content", label_col="label"):
        """
        df: pandas.DataFrame chứa dữ liệu
        stopwords: list stopwords cần loại bỏ (nếu có)
        text_col: tên cột chứa câu
        label_col: tên cột chứa nhãn
        """
        self.df = df.dropna(subset=[text_col, label_col])
        self.texts = self.df[text_col].astype(str)
        self.labels = self.df[label_col]
        preprocess = Preprocessor()
        self.stopwords = preprocess.stopwords
    
    def quality_check(self):
        """Kiểm tra dữ liệu null, trùng lặp"""
        nulls = self.df.isnull().sum()
        duplicates = self.df.duplicated().sum()
        print("Null values per column:")
        print(nulls)
        print(f"Number of duplicate rows: {duplicates}")
        return {"nulls": nulls, "duplicates": duplicates}

    # 1. Label distribution
    def label_distribution(self):
        dist = self.labels.value_counts(normalize=True) * 100
        print("Label distribution (%):")
        print(dist)
        sns.countplot(x=self.labels)
        plt.title("Label Distribution")
        plt.figure(figsize=(10, 10))
        plt.show()

    # 2. Sentence length statistics
    def sentence_length_stats(self):
        lengths = self.texts.apply(lambda x: len(x.split()))
        stats = lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        print("Sentence length statistics (word count):")
        print(stats)
        sns.histplot(lengths, bins=30, kde=True)
        plt.title("Sentence Length Distribution")
        plt.figure(figsize=(10, 10))
        plt.show()

    def vocab_stats(self, top_n=20, is_precessed=False):
        all_tokens = []
        for text in self.texts:
            if is_precessed:
                tokens = text.split()
            else:
                tokens = underthesea.word_tokenize(text)
            tokens = [t.lower() for t in tokens if re.match(r"\w+", t)]
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)
        vocab_size = len(counter)
        print(f"Vocabulary size: {vocab_size}")
        print(f"Top {top_n} most common words:")
        print(counter.most_common(top_n))

        # Vẽ biểu đồ
        common_words = counter.most_common(top_n)
        words, freqs = zip(*common_words)
        plt.figure(figsize=(max(10, top_n // 2), 10))
        plt.bar(words, freqs)
        plt.title(f"Top {top_n} most common words")
        plt.xticks(rotation=45)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    # 4. Stopword ratio
    def stopword_ratio(self, top_n=20, is_precessed=False):
        stop_count, total = 0, 0
        stop_tokens = []

        for text in self.texts:
            if is_precessed:
                tokens = text.split()
            else:
                tokens = underthesea.word_tokenize(text)
            tokens = [t.lower() for t in tokens if re.match(r"\w+", t)]
            total += len(tokens)
            for t in tokens:
                if t in self.stopwords:
                    stop_count += 1
                    stop_tokens.append(t)

        ratio = stop_count / total if total > 0 else 0
        print(f"Stopwords ratio: {ratio:.2%}")

        # Thống kê stopwords phổ biến nhất
        counter = Counter(stop_tokens)
        print(f"Top {top_n} most common stopwords:")
        common_stop = counter.most_common(top_n)
        for word, freq in common_stop:
            print(f"{word}: {freq}")

        # Vẽ biểu đồ
        if common_stop:
            words, freqs = zip(*common_stop)
            plt.figure(figsize=(max(10, top_n // 2), 10))
            plt.bar(words, freqs, color="orange")
            plt.title(f"Top {top_n} most common stopwords")
            plt.xticks(rotation=45)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
 
    # def top_ngrams(X_tfidf, vectorizer, top_n=20):
    #     freqs = zip(vectorizer.get_feature_names_out(), np.asarray(X_tfidf.sum(axis=0)).ravel())
    #     sorted_freqs = sorted(freqs, key=lambda x: -x[1])[:top_n]
    #     for word, freq in sorted_freqs:
    #         print(f"{word}: {freq:.2f}")
    #     return sorted_freqs

    # def sparsity(X_tfidf):
    #     nonzero = X_tfidf.count_nonzero()
    #     total = X_tfidf.shape[0] * X_tfidf.shape[1]
    #     sparsity = 1 - (nonzero / total)
    #     print(f"Sparsity: {sparsity:.2%}")
    #     return sparsity
    
    # def plot_top_tfidf_features(X_tfidf, vectorizer, top_n=20, figsize=(12,10)):
    #     # Tính tổng TF-IDF cho từng feature
    #     feature_sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
    #     features = vectorizer.get_feature_names_out()
    #     feature_freqs = list(zip(features, feature_sums))
        
    #     # Sắp xếp giảm dần
    #     sorted_features = sorted(feature_freqs, key=lambda x: -x[1])[:top_n]
        
    #     words, scores = zip(*sorted_features)
        
    #     # Vẽ biểu đồ cột
    #     plt.figure(figsize=figsize)
    #     plt.barh(range(len(words)), scores, color='skyblue')
    #     plt.yticks(range(len(words)), words)
    #     plt.gca().invert_yaxis()  # feature quan trọng nhất ở trên cùng
    #     plt.xlabel("Total TF-IDF score")
    #     plt.title(f"Top {top_n} TF-IDF features")
    #     plt.show()

