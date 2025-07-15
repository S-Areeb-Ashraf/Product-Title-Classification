
# Product Title Classification: Clarity and Conciseness

This project aims to assess and improve the **clarity** and **conciseness** of product titles based on their textual and categorical features. It utilizes feature engineering, ensemble machine learning, and a two-stage stacking model.

---

## Problem Statement

Given a product listing with title, description, and metadata (category, price, brand), the task is to predict:
- **Clarity Score**: How understandable and relevant the title is.
- **Conciseness Score**: How precise and succinct the title is.

These scores are useful for optimizing e-commerce listings and enhancing customer experience.

---

## Dataset Overview

Data is divided into:

- **Training Data**
  - `training/data_train.csv`
  - `training/clarity_train.labels`
  - `training/conciseness_train.labels`

- **Validation Data**
  - `validation/data_valid.csv`
  - `validation/clarity_valid.predict`
  - `validation/conciseness_valid.predict`

Each row contains:

```
market, prod_id, title, cat1, cat2, cat3, short_desc, price, brand
```

---

## Features Used

### Text Preprocessing
- Removed HTML tags from `short_desc`
- Cleaned special characters and lowercased text
- Combined `cat1`, `cat2`, `cat3` into a single cleaned category field

### TF-IDF Features
- Title (`1-2` grams, 1000 features)
- Short description (`1-2` grams, 1000 features)
- Combined category (`1` gram, 300 features)

### Numeric Features
- Title length (word count)
- Description length (word count)
- Number of HTML tags
- Cleaned price value (replaced `-1` with `0`)

---

## Model Pipeline

### Base Models (Stage 1)
Each of the following models was trained with **10-fold cross-validation**, repeated across **4 sets** (bagging):
- `LightGBMRegressor`
- `Ridge`
- `LogisticRegression`
- `SGDRegressor`

This resulted in 40 predictions per base model (10 folds × 4 seeds).

### Meta-Model (Stage 2: Stacking)
- An **XGBoostRegressor** was trained on the stacked base model predictions
- Separate regressors were trained for `clarity` and `conciseness`

---

## Results

### Average Base Model RMSEs
*(Sample output — update with actual values)*

| Model  | Clarity RMSE | Conciseness RMSE |
|--------|---------------|------------------|
| LGB    | 0.2890         | 0.2007           |
| Ridge  | 0.2864         | 0.1789           |
| LogReg | 0.3306         | 0.3180           |
| SGD    | 0.2794         | 0.1730           |

### Final Stacked Model Results

Final ensemble (XGBoost on top of base models):

```
Final RMSE (Clarity):     0.059980
Final RMSE (Conciseness): 0.064612
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code></summary>

```txt
pandas
numpy
scikit-learn
xgboost
lightgbm
scipy
```

</details>

### 2. Prepare directory structure

```
project_folder/
├── training/
│   ├── data_train.csv
│   ├── clarity_train.labels
│   ├── conciseness_train.labels
├── validation/
│   ├── data_valid.csv
│   ├── clarity_valid.predict
│   ├── conciseness_valid.predict
```

### 3. Run the script

```bash
python task.py
```

---

## Author

**Syed Areeb Ashraf**  
Computer Science Undergraduate | FAST-NUCES (Karachi)  
