from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from fields import *

rs = np.random.RandomState(12)


def columns_into_per_breast_rows(df, left_col, right_col, final_col):
    per_breast_df = pd.melt(df[FEATURE_COLS], value_vars=[left_col, right_col],
                            var_name=SIDE_COL, value_name=final_col, ignore_index=False)
    per_breast_df[SIDE_COL] = per_breast_df[SIDE_COL].str[0]
    return per_breast_df


def prepare_per_breast_df(df):
    sn_n_df = columns_into_per_breast_rows(df, L_SN_N_COL, R_SN_N_COL, SN_N_COL)
    n_imf_df = columns_into_per_breast_rows(df, L_N_IMF_COL,
                                            R_N_IMF_COL, N_IMF_COL).set_index(SIDE_COL, append=True)
    removed_df = columns_into_per_breast_rows(df, L_REMOVED_COL,
                                              R_REMOVED_COL, PER_BREAST_REMOVED_COL).set_index(SIDE_COL, append=True)

    patient_features_df = df.drop([L_SN_N_COL, L_N_IMF_COL, L_REMOVED_COL, R_SN_N_COL, R_N_IMF_COL, R_REMOVED_COL],
                                  axis=1).drop_duplicates()
    patient_features_df = patient_features_df.join(sn_n_df, how='outer', on=UNIQUE_ID_COL).set_index(SIDE_COL,
                                                                                                     append=True)
    patient_features_df = patient_features_df.join(n_imf_df, how='outer', on=[UNIQUE_ID_COL, SIDE_COL])
    patient_features_df = patient_features_df.join(removed_df, how='outer', on=[UNIQUE_ID_COL, SIDE_COL])
    return patient_features_df


def print_cross_validation_scores(cv_scores, cv, params):
    print(f"validation set results over {cv} splits")
    pprint(params)
    print("score\tmean\tstd")
    for k in cv_scores.keys():
        if k.startswith("test"):
            print(f"{k}\t{cv_scores[k].mean():.3f}\t{cv_scores[k].std()}")
    print()
    return


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


if __name__ == "__main__":
    breast_reduction_df = pd.read_csv("./data/RUMC Reduction Mammaplasties - MS 8-18-2021.csv")
    breast_reduction_df = breast_reduction_df.set_index(UNIQUE_ID_COL)
    data_df = prepare_per_breast_df(breast_reduction_df)
    print(f"Number of rows: {len(data_df)}")

    schnur_mae = mean_absolute_error(data_df[PER_BREAST_REMOVED_COL], data_df[SCHNUR_COL] / 2)
    schnur_mse = mean_squared_error(data_df[PER_BREAST_REMOVED_COL], data_df[SCHNUR_COL] / 2)
    print(f"Mean absolute error of schnur: {schnur_mae}\tMean Squared Error of schnur: {schnur_mse}")
    metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv = 5

    sample_mrns = data_df.index.values
    df_train, df_test, mrns_train, mrns_test = train_test_split(data_df, sample_mrns, test_size=0.33,
                                                                random_state=rs)
    test_schnur_mae = mean_absolute_error(df_test[PER_BREAST_REMOVED_COL], df_test[SCHNUR_COL] / 2)
    test_schnur_mse = mean_squared_error(df_test[PER_BREAST_REMOVED_COL], df_test[SCHNUR_COL] / 2)
    print(f"On the test set, schnur mae: {test_schnur_mae}\tschnur mse: {test_schnur_mse}")

    for use_schnur in [True, False]:

        ifc = TRAIN_FEATURE_COLS.copy()
        if use_schnur:
            ifc.append(SCHNUR_COL)

        X_train = df_train[ifc].values
        X_test = df_test[ifc].values
        y_train = df_train[PER_BREAST_REMOVED_COL].values
        y_test = df_test[PER_BREAST_REMOVED_COL].values
        print(f"Train set has {len(y_train)} samples, test set has {len(y_test)} samples.")

        best_models = {}
        for model_type in ['linear_regression', 'ridge_regression', 'SVR', 'RFR']:

            for standardize in [True, False]:

                if standardize:
                    pipe = [('scaler', StandardScaler())]
                else:
                    pipe = []

                if model_type == 'linear_regression':
                    reg = Pipeline(pipe + [(model_type, LinearRegression())])
                    cv_scores = cross_validate(reg, X_train, y_train, scoring=metrics, cv=cv, return_estimator=True)
                    params = {'model_type': model_type, 'use_schnur': use_schnur,
                              'standardize': standardize}
                    print_cross_validation_scores(cv_scores=cv_scores, cv=cv,
                                                  params=params)
                    best_models[model_type] = cv_scores['estimator'][cv_scores["test_neg_mean_absolute_error"].argmax()]


                elif model_type == 'ridge_regression':
                    best_mae = -float('inf')
                    best_mse = -float('inf')
                    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
                        reg = Pipeline(pipe + [(model_type, Ridge(alpha=alpha))])
                        cv_scores = cross_validate(reg, X_train, y_train, scoring=metrics, cv=cv, return_estimator=True)
                        params = {'model_type': model_type, 'use_schnur': use_schnur,
                                  'standardize': standardize, 'alpha': alpha}
                        # print_cross_validation_scores(cv_scores=cv_scores, cv=cv,
                        #                               params=params)

                        curr_mae = cv_scores['test_neg_mean_absolute_error'].mean()
                        if curr_mae > best_mae:
                            best_mae = curr_mae
                            best_mse = cv_scores['test_neg_mean_squared_error'].mean()
                            best_model_params = params
                            best_models[model_type] = cv_scores['estimator'][
                                cv_scores["test_neg_mean_absolute_error"].argmax()]
                    print(f"{best_model_params} produced MAE: {best_mae} and MSE: {best_mse}")
                elif model_type == 'SVR':
                    best_mae = -float('inf')
                    best_mse = -float('inf')
                    if standardize:
                        for kernel in ['linear', 'rbf', 'poly']:
                            for gamma in ['auto', 'scale']:
                                for C in [0.1, 0.5, 1.0, 2.0]:
                                    reg = Pipeline(pipe + [(model_type, SVR(kernel=kernel, gamma=gamma, C=C))])
                                    cv_scores = cross_validate(reg, X_train, y_train, scoring=metrics, cv=cv,
                                                               return_estimator=True)
                                    params = {'model_type': model_type,
                                              'use_schnur': use_schnur,
                                              'standardize': standardize,
                                              'kernel': kernel,
                                              'gamma': gamma,
                                              'C': C}
                                    # print_cross_validation_scores(cv_scores=cv_scores, cv=cv,
                                    #                               params=params)
                                    curr_mae = cv_scores['test_neg_mean_absolute_error'].mean()
                                    if curr_mae > best_mae:
                                        best_mae = curr_mae
                                        best_model_params = params
                                        best_mse = cv_scores['test_neg_mean_squared_error'].mean()
                                        best_models[model_type] = cv_scores['estimator'][
                                            cv_scores["test_neg_mean_absolute_error"].argmax()]
                        print(f"{best_model_params} produced MAE: {best_mae} and MSE: {best_mse}")

                elif model_type == 'RFR':
                    best_mae = -float('inf')
                    best_mse = -float('inf')
                    for n_estimators in [10, 50, 100, 500]:
                        for max_depth in [5, 10, 50, None]:
                            reg = Pipeline(pipe + [
                                (model_type, RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))])
                            cv_scores = cross_validate(reg, X_train, y_train, scoring=metrics, cv=cv,
                                                       return_estimator=True)
                            params = {'model_type': model_type,
                                      'use_schnur': use_schnur,
                                      'standardize': standardize,
                                      'n_estimators': n_estimators,
                                      'max_depth': max_depth}
                            # print_cross_validation_scores(cv_scores=cv_scores, cv=cv,
                            #                               params=params)
                            curr_mae = cv_scores['test_neg_mean_absolute_error'].mean()

                            if curr_mae > best_mae:
                                best_mae = curr_mae
                                best_model_params = params
                                best_mse = cv_scores['test_neg_mean_squared_error'].mean()
                                best_models[model_type] = cv_scores['estimator'][
                                    cv_scores["test_neg_mean_absolute_error"].argmax()]

                    print(f"{best_model_params} produced MAE: {best_mae} and MSE: {best_mse}")
                    importances = best_models[model_type][model_type].feature_importances_
                    print(f"RFR with {best_model_params} had variable importances: {importances}, values respectively: {ifc}")

        for model_type, model in best_models.items():
            y_pred = model.predict(X_test)
            error = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{model_type} got mean absolute error: {error} and mean square error: {mse} on test set.")

            fig = plt.figure(figsize=(11, 5))

            plt.plot(y_pred, 'bo', label='prediction')
            plt.plot(y_test, 'ro', label='actual')
            # adjust bottom of plot to remove cutoff
            plt.gcf().subplots_adjust(bottom=0.15)
            # Axis Label, title, and legend
            plt.xlabel('Patient #')
            plt.ylabel('Tissue Removed (g)')
            plt.title('Actual and Predicted Values')
            # plt.legend()
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='upper right')

            plt.tick_params(axis='x', which='major', labelsize=10)
            plt.savefig(f"./output_figures/actual_vs_expected_model_type={model_type}_useschnur={use_schnur}.png")
            # plt.show()
