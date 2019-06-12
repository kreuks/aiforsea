import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import Booster


class Modeler:
    def fit(self, X_train, y_train, param, *args, **kwargs):
        pass

    def predict(self, X_test):
        pass

    def pr_auc_score(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        return pr_auc

    def roc_auc_score(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)


class XGBoostModeler(Modeler):
    def __init__(self):
        self.model: Booster = None

    def fit(self, dtrain, dval, param, *args, **kwargs):
        param['max_depth'] = int(param['max_depth'])
        self.model = xgb.train(dtrain=dtrain,
                               num_boost_round=kwargs['num_boost_round'],
                               evals=kwargs['watchlist'],
                               early_stopping_rounds=kwargs['early_stopping_rounds'],
                               params=param)

    def predict(self, dtest):
        return self.model.predict(dtest)


class LGBModeler(Modeler):
    def fit(self, X_train, y_train, param, *args, **kwargs):
        pass

    def predict(self, X_test):
        pass


class Optimizer:
    def __init__(self, X, y):
        self._modeler = XGBoostModeler()
        self.X = X
        self.y = y

    def scorer(self, params):
        num_round = int(params['n_estimators'])
        del params['n_estimators']

        n_fold = 5
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

        total_roc_auc = 0
        for fold_n, (train_index, valid_index) in enumerate(folds.split(self.X)):
            X_train, X_valid = self.X.iloc[train_index], self.X.iloc[valid_index]
            y_train, y_valid = self.y.iloc[train_index], self.y.iloc[valid_index]

            dtrain = xgb.DMatrix(data=X_train, label=y_train)
            dval = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(dtrain, 'train'), (dval, 'valid_data')]

            self._modeler.fit(dtrain=dtrain, dval=dval, param=params, watchlist=watchlist, num_boost_round=num_round,
                              early_stopping_rounds=15)

            y_pred_valid = self._modeler.predict(dval)

            total_roc_auc += (self._modeler.roc_auc_score(y_valid, y_pred_valid) / n_fold)

        loss = 1 - total_roc_auc
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self, trials):
        space = {'n_estimators': 20,
                 'booster': 'dart',
                 'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
                 'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),
                 'rate_drop': hp.uniform('rate_drop', 0, 1),
                 'skip_drop': hp.uniform('skip_drop', 0, 1),
                 # 'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
                 'eta': hp.loguniform('eta', -3.5, -1.5),

                 'max_depth': hp.choice('max_depth', np.arange(1, 30, dtype=int)),

                 'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                 # 'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                 'subsample': hp.uniform('subsample', 0.5, 1),
                 'alpha': hp.uniform('alpha', 0, 0.5),
                 'lambda': hp.uniform('lambda', 1e-4, 1),
                 'scale_pos_weight_multiplier': hp.choice('scale_pos_weight_multiplier',
                                                          [0, 0.1, 1., 10, 100, 1e3, 1e4, 1e5, 1e6]),

                 'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                 'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                 # 'num_class': 2,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'nthread': 0,
                 'silent': 1}

        best = fmin(self.scorer, space, algo=tpe.suggest, trials=trials, max_evals=1)

        for k, v in best.items():
            space[k] = v

        with open('safety/models/xgboost/best/hyperparameter.json', 'w') as f:
            json.dump(space, f)

    def run(self):
        os.makedirs('safety/models/xgboost/best/', exist_ok=True)
        trials = Trials()
        self.optimize(trials)


if __name__ == '__main__':
    df = pd.read_csv('coba.csv')
    col = df.columns
    # col = [x for x in col if not x.startswith('label_')]
    # df = df[col]
    df_features = df[col[1:-1]]
    df_label = df[col[-1]]
    opt = Optimizer(df_features, df_label)
    opt.run()
