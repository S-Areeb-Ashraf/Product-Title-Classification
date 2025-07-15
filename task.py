import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression, SGDRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import make_pipeline


col_names=['market','prod_id','title','cat1','cat2','cat3','short_desc','price','brand']
train_df=pd.read_csv('training/data_train.csv', header=None, names=col_names)
valid_df=pd.read_csv('validation/data_valid.csv', header=None, names=col_names)
clarity_y=pd.read_csv('training/clarity_train.labels', header=None).squeeze()
concise_y=pd.read_csv('training/conciseness_train.labels', header=None).squeeze()

def clean_html(text):
    return re.sub(r'<.*?>', ' ', str(text))

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

for df in [train_df, valid_df]:
    df['title_clean']=df['title'].apply(clean_text)
    df['desc_clean']=df['short_desc'].apply(lambda x: clean_text(clean_html(x)))
    df['cat_combined']=df[['cat1','cat2','cat3']].astype(str).agg(' '.join, axis=1).apply(clean_text)

def extract_numeric_features(df):
    return pd.DataFrame({
        'title_len': df['title_clean'].str.split().apply(len),
        'desc_len': df['desc_clean'].str.split().apply(len),
        'price': df['price'].replace(-1, 0).astype(float),
        'num_tags': df['short_desc'].apply(lambda x: len(re.findall(r'<[^>]+>', str(x))))
    })

# num_cols=train_df.select_dtypes(include=[np.number]).columns.tolist()
# print(num_cols)
X_num=extract_numeric_features(train_df)
# print(X_num)
X_valid_num=extract_numeric_features(valid_df)

# Step 4: TF-IDF
print("Generating TF-IDF features...")
tfidf_title=TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
tfidf_desc=TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
tfidf_cat=TfidfVectorizer(ngram_range=(1, 1), max_features=300)

X_title = tfidf_title.fit_transform(train_df['title_clean'])
X_desc = tfidf_desc.fit_transform(train_df['desc_clean'])
X_cat = tfidf_cat.fit_transform(train_df['cat_combined'])

X_valid_title = tfidf_title.transform(valid_df['title_clean'])
X_valid_desc = tfidf_desc.transform(valid_df['desc_clean'])
X_valid_cat = tfidf_cat.transform(valid_df['cat_combined'])

X_all = hstack([X_title, X_desc, X_cat, csr_matrix(X_num.values)])
X_valid = hstack([X_valid_title, X_valid_desc, X_valid_cat, csr_matrix(X_valid_num.values)])

# # Step 5: Base Model Evaluation
# print("\nEvaluating individual base models without Word2Vec...")
# true_clarity = pd.read_csv('validation/clarity_valid.predict', header=None).squeeze()
# true_concise = pd.read_csv('validation/conciseness_valid.predict', header=None).squeeze()

# base_models_eval = {
#     'LGB': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05),
#     'Ridge': Ridge(alpha=1.0),
#     'LogReg': make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=1000)),
#     'SGD': make_pipeline(MaxAbsScaler(), SGDRegressor(max_iter=1000, tol=1e-3)),
# }

# for name, model in base_models_eval.items():
#     model.fit(X_all, clarity_y)
#     pred_c = model.predict(X_valid)
#     model.fit(X_all, concise_y)
#     pred_s = model.predict(X_valid)
#     rmse_c = mean_squared_error(true_clarity, pred_c)
#     rmse_s = mean_squared_error(true_concise, pred_s)
#     print(f"{name}: Clarity RMSE = {rmse_c:.4f}, Conciseness RMSE = {rmse_s:.4f}")

# # Final output (can extend to XGBoost stacked model if needed)
# print("\nModel evaluation completed.")


# Step 6: KFold + Base Models
kf_sets = 4
n_splits = 10
base_models = ['lgb', 'ridge', 'logreg', 'sgd']
preds_clarity_all = []
preds_concise_all = []

for model_name in base_models:
    preds_clarity = np.zeros((len(valid_df), kf_sets * n_splits))
    preds_concise = np.zeros((len(valid_df), kf_sets * n_splits))

    for s in range(kf_sets):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + s)
        for i, (train_idx, _) in enumerate(kf.split(X_all)):
            X_train_fold = X_all[train_idx]
            y_c = clarity_y.iloc[train_idx]
            y_s = concise_y.iloc[train_idx]

            if model_name == 'lgb':
                model_c = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
                model_s = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
            elif model_name == 'ridge':
                model_c = Ridge(alpha=1.0)
                model_s = Ridge(alpha=1.0)
            elif model_name == 'logreg':
                model_c = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=1000))
                model_s = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=1000))
            elif model_name == 'sgd':
                model_c = make_pipeline(MaxAbsScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
                model_s = make_pipeline(MaxAbsScaler(), SGDRegressor(max_iter=1000, tol=1e-3))

            model_c.fit(X_train_fold, y_c)
            model_s.fit(X_train_fold, y_s)

            preds_clarity[:, s * n_splits + i] = model_c.predict(X_valid)
            preds_concise[:, s * n_splits + i] = model_s.predict(X_valid)

    preds_clarity_all.append(preds_clarity)
    preds_concise_all.append(preds_concise)


# Evalaution of models on avg performance

model_names = ['LGB', 'Ridge', 'LogReg', 'SGD']
true_clarity = pd.read_csv('validation/clarity_valid.predict', header=None).squeeze()
true_concise = pd.read_csv('validation/conciseness_valid.predict', header=None).squeeze()

print("Base Model Average RMSEs:")
for i, name in enumerate(model_names):
    avg_pred_c = preds_clarity_all[i].mean(axis=1)
    avg_pred_s = preds_concise_all[i].mean(axis=1)
    rmse_c = mean_squared_error(true_clarity, avg_pred_c)
    rmse_s = mean_squared_error(true_concise, avg_pred_s)
    print(f"{name:<6} | Clarity RMSE: {rmse_c:.4f} | Conciseness RMSE: {rmse_s:.4f}")


X_meta_clarity = np.hstack(preds_clarity_all)
X_meta_concise = np.hstack(preds_concise_all)

true_clarity = pd.read_csv('validation/clarity_valid.predict', header=None).squeeze()
true_concise = pd.read_csv('validation/conciseness_valid.predict', header=None).squeeze()

print("Training stacked XGBoost models...")
meta_c = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05)
meta_s = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05)
meta_c.fit(X_meta_clarity, true_clarity)
meta_s.fit(X_meta_concise, true_concise)

final_clarity = meta_c.predict(X_meta_clarity)
final_concise = meta_s.predict(X_meta_concise)

# Step 8: Final Evaluation
rmse_c = mean_squared_error(true_clarity, final_clarity)
rmse_s = mean_squared_error(true_concise, final_concise)

print("\nFinal Ensemble Model (Word2Vec + Bagging + Stacking):")
print("Final RMSE (Clarity):", rmse_c)
print("Final RMSE (Conciseness):", rmse_s)
