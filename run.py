import pandas as pd
import os
import gc
import lightgbm as lgb

from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, precision_score, recall_score
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')


"""
读取数据集
"""
train_basic = pd.read_csv('data_jk/train_basic_info.csv')
train_invest = pd.read_csv('data_jk/train_investor_info.csv')
train_tax_payment = pd.read_csv('data_jk/train_tax_payment_.csv')
train_tax_return = pd.read_csv('data_jk/train_tax_return_.csv')
train_label = pd.read_csv('data_jk/train_label.csv')
train_label.rename(columns={'SHXYDM': 'ID'}, inplace=True)

test_basic = pd.read_csv('data_jk/test_basic_info.csv')
test_invest = pd.read_csv('data_jk/test_investor_info.csv')
test_tax_payment = pd.read_csv('data_jk/test_tax_payment_.csv')
test_tax_return = pd.read_csv('data_jk/test_tax_return_.csv')

train = pd.merge(train_basic, train_label, on='ID', how='left')
train['is_train'] = 1
data = pd.concat([train, test_basic]).reset_index(drop=True)



"""
基础数据处理
"""
HYML_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
             'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18}
data['HYML_DM'] = data['HYML_DM'].map(HYML_dict)

le_encoder = LabelEncoder()
for categorical_feature in tqdm(
        ['code_enterprise_ratal_classes', 'code_enterprise_registration', 'HY_DM', 'HYZL_DM', 'HYDL_DM',
         ]):
    data[categorical_feature] = le_encoder.fit_transform(data[categorical_feature])

data['enterprise_opening_date'] = pd.to_datetime(data['enterprise_opening_date'])
data['enterprise_opening_date_year'] = data['enterprise_opening_date'].dt.year


"""
简单特征工程
"""
data_tax_return = pd.concat([train_tax_return, test_tax_return]).reset_index(drop=True)
data_tax_return.sort_values('tax_return_end')
data_tax_return.drop_duplicates('ID', inplace=True, )

data_tax_return['tax_return_end'] = pd.to_datetime(data_tax_return['tax_return_end'])
data_tax_return['tax_return_begin'] = pd.to_datetime(data_tax_return['tax_return_begin'])
data_tax_return['tax_return_end_month'] = data_tax_return['tax_return_end'].dt.month

data = data.merge(data_tax_return, on=['ID'], how='left')

for col in ['code_account', 'code_item']:
    data[col + '_base_count'] = data[col].map(data[col].value_counts())

data_invest = pd.concat([train_invest, test_invest]).reset_index(drop=True)

tmp = data_invest.groupby(['ID'])['investor_rate'].agg([
    ('max_investor_rate', 'max'),
    ('min_investor_rate', 'min'),
    ('mean_investor_rate', 'mean'),
    ('std_investor_rate', 'std'),
]).reset_index()
data = data.merge(tmp, on=['ID'], how='left')

tmp = data_invest.groupby(['ID'])['investor_amount'].agg([
    ('max_investor_amount', 'max'),
    ('min_investor_amount', 'min'),
    ('mean_investor_amount', 'mean'),
    ('std_investor_amount', 'std'),
]).reset_index()
data = data.merge(tmp, on=['ID'], how='left')

"""
训练lgb模型
"""

def train_model(X_train, X_test, features, y, save_model=False):

    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': 2021,
        'bagging_seed': 2021,
        'feature_fraction_seed': 2021,
        'early_stopping_rounds': 100,
    }
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros((len(X_test)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,

        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5
        if save_model:
            clf.save_model(f'model_{fold_}.txt')

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

    return feat_imp_df, oof_lgb, predictions_lgb


train = data[~data['is_train'].isna()].reset_index(drop=True)
test = data[data['is_train'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['ID', 'enterprise_opening_date', 'label', 'is_train',
                                                  'tax_return_date', 'tax_return_deadline', 'tax_return_begin',
                                                  'tax_return_end',

                                                  ]]
y = train['label']

feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y)


"""
特征筛选并重新训练
"""
feat_imp_df = feat_imp_df.sort_values('imp')
features = feat_imp_df[feat_imp_df['imp'] > 5]['feat']
_, oof_lgb, predictions_lgb = train_model(train, test, features, y)


"""
生成提交文件
"""
test['predict_prob'] = predictions_lgb
test[['ID', 'predict_prob']].to_csv('submission.csv', index=False)
