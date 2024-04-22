import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_csv('prepro_data.csv')
df = df.dropna()

x = df['prepro'].copy()
y = df['label'].copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# BoW 임베딩
bow_vectorizer = CountVectorizer(analyzer='word', max_features=3000)
x_bow_train = bow_vectorizer.fit_transform(x_train)
x_bow_val = bow_vectorizer.transform(x_val)
x_bow_test = bow_vectorizer.transform(x_test)

# TF-IDF 임베딩
tfidf_vectorizer = TfidfVectorizer(min_df=0.0, analyzer='word', ngram_range=(1,3), max_features=3000)
x_tfidf_train = tfidf_vectorizer.fit_transform(x_train)
x_tfidf_val = tfidf_vectorizer.transform(x_val)
x_tfidf_test = tfidf_vectorizer.transform(x_test)

# Word2Vec 임베딩
sentences_train = [sentence.split() for sentence in x_train]
sentences_val = [sentence.split() for sentence in x_val]
sentences_test = [sentence.split() for sentence in x_test]
w2v_model = word2vec.Word2Vec(sentences_train, vector_size=500, min_count=10, window=10)
x_w2v_train = np.array([np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv] or [np.zeros(500)], axis=0) for words in sentences_train])
x_w2v_val = np.array([np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv] or [np.zeros(500)], axis=0) for words in sentences_val])
x_w2v_test = np.array([np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv] or [np.zeros(500)], axis=0) for words in sentences_test])

# 모델 정의 및 평가 함수
def evaluate_model(model, params, x_train, y_train, x_val, y_val, x_test, y_test, model_name, vector_name):

    if issparse(x_train):
        x_train = x_train.toarray()
    if issparse(x_val):
        x_val = x_val.toarray()
    if issparse(x_test):
        x_test = x_test.toarray() # 강충원 열심히해


    grid = GridSearchCV(model, params, cv=5, refit=True, return_train_score=True)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    pred_y_train = best_model.predict(x_train)
    train_accuracy = accuracy_score(y_train, pred_y_train)

    pred_y_val = best_model.predict(x_val)
    val_accuracy = accuracy_score(y_val, pred_y_val)

    pred_y_test = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, pred_y_test)

    print(f"{model_name} ({vector_name}) train accuracy: {train_accuracy:.4f}")
    print(f"{model_name} ({vector_name}) validation accuracy: {val_accuracy:.4f}")
    print(f"{model_name} ({vector_name}) test accuracy: {test_accuracy:.4f}")
    print(f"{model_name} ({vector_name}) best parameters: {grid.best_params_}")
    joblib.dump(best_model, f"{model_name}_{vector_name}_best_model.joblib")

# 모델과 파라미터 설정
models = [
    #(LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=42), {"max_iter": [1000, 2000, 3000], "C": [0.1, 1, 10], "penalty": ['l2']}),
    #(GaussianNB(), {'var_smoothing': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}),
    (SVC(class_weight='balanced', random_state=42), {'C': [1e-1, 1, 10], 'gamma': ['scale', 1e-3, 1e-1, 1], 'kernel': ['rbf']}),
    (RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42), {'n_estimators': [100, 150, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5], 'min_impurity_decrease': [0, 1e-1, 1e-2]}),
    (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, device = 'gpu', random_state=42), {'gamma': [0, 1e-1, 1e-2], 'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3, 0.5], 'subsample': [0.6, 0.8, 1], 'colsample_bytree': [0.6, 0.8, 1]})
]

# 각 임베딩과 모델에 대한 평가 실행
for model, params in models:
    model_name = model.__class__.__name__
    evaluate_model(model, params, x_bow_train, y_train, x_bow_val, y_val, x_bow_test, y_test, model_name, "BoW")
    evaluate_model(model, params, x_tfidf_train, y_train, x_tfidf_val, y_val, x_tfidf_test, y_test, model_name, "TF-IDF")
    evaluate_model(model, params, x_w2v_train, y_train, x_w2v_val, y_val, x_w2v_test, y_test, model_name, "Word2Vec")
