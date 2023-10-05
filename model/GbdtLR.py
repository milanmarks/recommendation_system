import pandas as pd
import lightgbm as lgb
import category_encoders as ce
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Training samples path, change to your local path
training_samples_file_path = "../data/trainingSamples.csv"
# Test samples path, change to your local path
test_samples_file_path = "../data/testSamples.csv"

train_dataset = pd.read_csv(training_samples_file_path)
test_dataset = pd.read_csv(test_samples_file_path)
X_train = train_dataset.drop(["rating", "label"], axis=1)
y_train = train_dataset["label"]
X_test = test_dataset.drop(["rating", "label"], axis=1)
y_test = test_dataset["label"]

cate_cols = ["movieGenre1", "movieGenre2", "movieGenre3", "userGenre1", "userGenre2", "userGenre3", "userGenre4",
             "userGenre5"]
ord_encoder = ce.ordinal.OrdinalEncoder(cols=cate_cols)
X_train = ord_encoder.fit_transform(X_train)
X_test = ord_encoder.transform(X_test)

# specify your configurations as a dict
params = {
    "task": "train",
    "boosting_type": "gbdt",
    "num_class": 1,
    "objective": "binary",
    "metric": {"auc", "rmse"},
    "num_leaves": 64,
    "min_data": 20,
    "boost_from_average": True,
    # set it according to your cpu cores.
    "num_threads": 8,
    "feature_fraction": 0.8,
    "learning_rate": 0.15,
}

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, params=params, categorical_feature=cate_cols)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=cate_cols)
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round=100,
                      valid_sets=lgb_train,
                      categorical_feature=cate_cols,
                      callbacks=[lgb.early_stopping(20)])

test_preds = lgb_model.predict(X_test)
auc = roc_auc_score(np.asarray(y_test), np.asarray(test_preds))
pr_auc = average_precision_score(np.asarray(y_test), np.asarray(test_preds))
res_basic = {"auc": auc, "pr_auc": pr_auc}
print(res_basic)

