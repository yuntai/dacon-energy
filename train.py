import numpy as np
import xgboost as xgb
from sklearn.metrics import make_scorer
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score, KFold, cross_validate

#mape_scorer = make_scorer(mape, greater_is_better=False)
# run XGBoost algorithm with hyperparameters optimization
def train_xgb(params, X_train, y_train, cv, scorer='neg_mean_squared_error', seed=42):
    """
    Train XGBoost regressor using the parameters given as input. The model
    is validated using standard cross validation technique adapted for time series
    data. This function returns a friendly output for the hyperopt parameter optimization
    module.

    Parameters
    ----------
    params: dict with the parameters of the XGBoost regressor. For complete list see:
            https://xgboost.readthedocs.io/en/latest/parameter.html
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets

    Returns
    -------
    dict with keys 'model' for the trained model, 
    'status' containing the hyperopt,
    status string,
    and 'loss' with the RMSE obtained from cross-validation
    """

    n_estimators = int(params["n_estimators"])
    max_depth= int(params["max_depth"])

    try:
        model = xgb.XGBRegressor(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 learning_rate=params["learning_rate"],
                                 subsample=params["subsample"], seed=seed)

        #result = model.fit(X_train,
        #                   y_train.values.ravel(),
        #                   eval_set=[(X_train, y_train.values.ravel())],
        #                   early_stopping_rounds=50,
        #                   verbose=False)

        # cross validate using the right iterator for time series
        #cv_space = KFold(n_splits=n_splits)

        cv_score = cross_validate(
            model,
            X_train, y_train.values.ravel(),
            cv=cv,
            scoring=scorer,
            return_estimator=True
        )

        avg_score = np.abs(np.mean(np.array(cv_score['test_score'])))
        return {
            "loss": avg_score,
            "status": STATUS_OK,
            "models": cv_score['estimator']
        }

    except ValueError as ex:
        return {
            "error": ex,
            "status": STATUS_FAIL
        }

def optimize_xgb(X_train, y_train, n_splits=5, max_evals=10, cv=None, scorer='neg_mean_squared_error', seed=42):
    """
    Run Bayesan optimization to find the optimal XGBoost algorithm
    hyperparameters.

    Parameters
    ----------
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets
    max_evals: the maximum number of iterations in the Bayesian optimization method

    Returns
    -------
    best: dict with the best parameters obtained
    trials: a list of hyperopt Trials objects with the history of the optimization
    """
    assert cv is not None

    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
        "max_depth": hp.quniform("max_depth", 1, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", -5, 1),
        "subsample": hp.uniform("subsample", 0.8, 1),
        "gamma": hp.quniform("gamma", 0, 100, 1)
    }

    objective_fn = partial(train_xgb,
                           X_train=X_train, y_train=y_train, 
                           scorer=scorer, 
                           cv=cv,
                           seed=seed)

    trials = Trials()
    best = fmin(fn=objective_fn,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    # evaluate the best model on the test set
    return best, trials
