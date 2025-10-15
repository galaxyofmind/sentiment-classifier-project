from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class Vectorizer_TFIDF:
    def __init__(self, ngram_range=(1,2), min_df=31, max_features=50000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features
        )
    
    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()
    
    def transform(self, X):
        return self.vectorizer.transform(X).toarray()
    
    def info(self, X):
        print("Kích thước ma trận TF-IDF:", X.shape)
        #print("Số features:", len(self.vectorizer.get_feature_names_out()))
        #print("Ví dụ 10 features đầu:", self.vectorizer.get_feature_names_out()[:10])
