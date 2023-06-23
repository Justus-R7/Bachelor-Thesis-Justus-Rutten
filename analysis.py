from os import listdir

print("done")
import pandas as pd
from time import perf_counter

pd.options.display.max_columns = 6

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

from sklearn import model_selection
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

from sklearn.impute import KNNImputer
from collections import Counter
import numpy as np
from numpy import absolute


from xgboost import XGBRegressor
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, chi2, r_regression

from sklearn.preprocessing import scale

sns.set(style='darkgrid', context='talk', palette='rainbow')


data_historie = pd.read_excel("Test sets/Datasets/data_historie_language_full.xlsx")
data_actueel = pd.read_excel("Test sets/Datasets/data_actueel_language_full.xlsx")

data_historie.drop(columns=data_historie.columns[0], axis=1, inplace=True)
data_historie.drop(index=0, inplace=True)
data_historie.drop(['Current', 'L-year', '2-years'], axis=1, inplace=True)

data_actueel.drop(columns=data_actueel.columns[0], axis=1, inplace=True)
data_actueel.drop(index=0, inplace=True)
data_actueel.drop(['class'], axis=1, inplace=True)


def get_x_and_y_sets(course, data, filename, imputation, classification, pass_value, percentage_full_row,
                     percentage_full_column, g):
    # Select the correct data set. Final implementation only uses data = pd.read_excel(filename)
    if data == 0:
        data = data_historie
    elif data == 1:
        data = data_actueel
    elif data == 2:
        data = pd.concat([data_historie, data_actueel], axis=1)
    else:
        data = pd.read_excel(filename)
        data.drop(columns=data.columns[0], axis=1, inplace=True)

    # change the course to the correct number, because we loop through each course multiple times
    course_correct = course[0]
    course_correct = course_correct[:-1]
    course_correct = course_correct + str(g)
    course[0] = course_correct

    # sort dataset from low grades to high grades. This was used to help understand the data early on in combination with visualisation.
    if classification == 0:
        data.sort_values(by=[course[0]], inplace=True)

    # Drop all rows that do not have a grade in the course
    data_course = data
    for c in course:
        data_course = data_course[data_course[c].notna()]

    # drop all grades that are chronologically after this one
    grade_number = g
    clean_data = []
    for i in range(0, int(grade_number) + 1):
        clean_data = clean_data + [col for col in data_course if col.endswith(str(i))]
    clean_data = data_course[clean_data]
    data_course = clean_data

    # flip percentage row and columns, so the number correctly represents what we need.
    percentage_full_row = 100 - percentage_full_row
    percentage_full_column = 100 - percentage_full_column

    # drop all columns that do not meet de % requirement
    droplist = []
    for column in data_course:
        if data_course[column].isnull().sum() / data_course.shape[0] > percentage_full_column * 0.01:
            droplist.append(column)

    data_course = data_course.drop(columns=droplist)

    droplistrow = []
    for index, row in data_course.iterrows():
        if row.isnull().sum() / data_course.shape[1] > percentage_full_row * 0.01:
            droplistrow.append(index)

    data_course = data_course.drop(index=droplistrow)

    x = data_course.loc[:, :]
    # Drop the course from the input set x
    x.drop(course, axis=1, inplace=True)

    # impute if necessary
    if imputation == 1:
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

    # transformation was used for pca. not in final implemenation.
    # X_reduced = pca.fit_transform(scale(x))
    # X = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
    X = x

    # Save target columns, looping was for multioutput. Not in final implementation but looping was never removed.
    y = pd.DataFrame()
    for c in course:
        y[c] = data_course[c]
    if classification == 1:
        y = y.where(y > pass_value, 0)
        y = y.where(y < pass_value, 1)

    if classification == 2:
        y = y.where(y > pass_value, 0)
        y = y.where(y <= 7, 2)
        y = y.where(y < pass_value, 1)

    return X, y


def create_data_set(course, X, y, stratify):
    if stratify == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=42)


    # printing was originally used to help understand the data.

    # print("========== Training data ========== ")
    # print(f"Features: {X_train.shape} | Target:{y_train.shape}")
    # print("========== Validate data ========== ")
    # print(f"Features: {X_validate.shape} | Target:{y_validate.shape}")
    # print("========== Test data ========== ")
    # print(f"Features: {X_test.shape} | Target:{y_test.shape}")
    return X_train, X_test, y_train, y_test


def create_model(model, X_train, X_test, y_train, y_test, X, y, summised_score, multioutput):
    # set up cross-validation and model
    if len(y) >= 300:
        # print('over 300')
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    elif len(y) >= 100:
        # print('sub 300 but over 100')
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    elif len(y) >= 50:
        # print('sub 100')
        cv = RepeatedKFold(n_splits=len(y), n_repeats=3, random_state=1)
    else:
        # print('under 50 students')
        cv = RepeatedKFold(n_splits=len(y), n_repeats=3, random_state=1)

    scores = cross_val_score(model, X, y.iloc[:, 0].ravel(), scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = absolute(scores)

    # fit differently based on multioutput. Not in final implementation.
    if multioutput == 0:
        model.fit(X_train, y_train.iloc[:, 0].ravel())
        score = model.score(X_test, y_test.iloc[:, 0].ravel())
        # print(score)
    else:
        model.fit(X_train, y_train)
        # score = model.score(X_test, y_test)

    summised_score.append(scores.mean())
    summised_score.append(scores.std())
    summised_score.append(score)
    return model, summised_score

# This method was used to understand the best way to visualise classification results
def visualise(model, X_train, X_test, y_train, y_test, X, y):
    if multioutput == 0:
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        # Visualisation
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
        vis.plot()
        plt.show()
        model_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
        rfc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=model_disp.ax_)
        rfc_disp.figure_.suptitle("ROC curve comparison")
        plt.show()
        print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        # roc_auc_score(y_test, model.decision_function(X_test))
    else:
        y_pred = model.predict(X_test)
        print(y_pred)
        print(confusion_matrix(y_test.iloc[:, 0], y_pred))
        print(classification_report(y_test.iloc[:, 0], y_pred))
        # Visualisation
        conf_matrix = confusion_matrix(y_true=y_test.iloc[:, 0], y_pred=y_pred)
        vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
        vis.plot()
        plt.show()
        model_disp = RocCurveDisplay.from_estimator(model, X_test, y_test.iloc[:, 0])
        rfc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test.iloc[:, 0], ax=model_disp.ax_)
        rfc_disp.figure_.suptitle("ROC curve comparison")
        plt.show()
        print(roc_auc_score(y_test.iloc[:, 0], model.predict_proba(X_test)[:, 1]))
        # roc_auc_score(y_test.iloc[:, 1], model.decision_function(X_test))

        y_pred = model.predict_proba(X_test)
        print(y_pred)
        roc_auc_score(y_test, y_pred, multi_class='ovr')
        roc_auc_score(y_test, y_pred, average=None)
        print('Roc_auc scores:')
        print(roc_auc_score(y_test, y_pred, multi_class='ovr'))
        print(roc_auc_score(y_test, y_pred, average=None, multi_class='ovr'))

    # mapie = MapieClassifier(estimator=model, cv='prefit').fit(X_train, y_train)

    # NOTE HERE I DELETE X

    # _, y_pi_mapie = mapie.predict(X_test, alpha=0.05)
    # print(y_pi_mapie[:, :, 0])


# This method was used to understand the best way to visualise regression results
def visualise_regression(X, y, m):
    alpha = [0.05, 0.2]
    mapie = MapieRegressor(estimator=m, cv='prefit', method='plus')
    mapie = mapie.fit(X, y.iloc[:, 0].ravel())
    y_pred, y_pis = mapie.predict(X, alpha=alpha)
    y_pred_model = m.predict(X)
    coverage_scores = [
        regression_coverage_score(y, y_pis[:, 0, i], y_pis[:, 1, i])
        for i, _ in enumerate(alpha)
    ]

    # NOTE HERE I DELETE X
    X_data = X
    X = [[i] for i in range(len(X))]
    X = np.asarray(X)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, color="#d62728", alpha=0.6, marker='+')
    plt.scatter(X, y_pred, color="#7f7f7f", alpha=0.3, marker='x')
    order = np.argsort(X[:, 0])
    plt.plot(X[order], y_pis[order][:, 0, 1], color="C1", ls="--")
    plt.plot(X[order], y_pis[order][:, 1, 1], color="C1", ls="--")
    plt.fill_between(
        X[order].ravel(),
        y_pis[order][:, 0, 0].ravel(),
        y_pis[order][:, 1, 0].ravel(),
        alpha=0.2
    )
    plt.title(
        f"Target and effective coverages for "
        f"alpha={alpha[0]:.2f}: ({1 - alpha[0]:.3f}, {coverage_scores[0]:.3f})\n"
        f"Target and effective coverages for "
        f"alpha={alpha[1]:.2f}: ({1 - alpha[1]:.3f}, {coverage_scores[1]:.3f})"
    )

    plt.show()

    # calculate what percentage of the bar is taken
    for i in alpha:
        mapie = mapie.fit(X_data, y.iloc[:, 0].ravel())
        y_pred, y_pis = mapie.predict(X_data, alpha=i)
        width = 0
        for j in range(X_data.shape[0]):
            width = width + (y_pis[j, 1] - y_pis[j, 0]) / 9
        width = width / X_data.shape[0]
        print(width * 100)


# PCA was tried but not fully implemented
def p_c_a(X, y, m):
    # scale predictor variables
    X_reduced = pca.fit_transform(scale(X))

    # define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    mse = []

    # Calculate MSE with only the intercept
    score = -1 * model_selection.cross_val_score(m,
                                                 np.ones((len(X_reduced), 1)), y, cv=cv,
                                                 scoring='neg_mean_absolute_error').mean()
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, len(X.columns)):
        score = -1 * model_selection.cross_val_score(m,
                                                     X_reduced[:, :i], y, cv=cv,
                                                     scoring='neg_mean_absolute_error').mean()
        mse.append(score)

    # Plot cross-validation results
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MAE')
    plt.title('grade prediction')
    plt.show()
    mse.pop(0)
    best_amount_of_components = mse.index(min(mse))
    return best_amount_of_components


def test_regression(directory, filename_format, courses):
    # This method is a culmination of many smaller methods, these were originally split but as time went on it proved (momentarily) easier to create this frankenstein.
    # This is the main thing that I would have done differently, I should have stuck to separate methods.

    print('regression')
    final_scores = pd.DataFrame()
    multioutput = 0
    courses_english = ['Dutch', 'English', 'Mathematics']
    most_influencial_courses = []
    for f in listdir(directory):
        # The next 22 lines of code set up the parameters for regression models.

        alpha = [0.05, 0.1, 0.2, 0.3, 0.4]
        # data = 0 means just historie, 1 = actueel, 2 is both, 3 is own dataset
        data = 3
        filename = f'{filename_format}\\{f}'
        current_class = f.replace("classes_actueel_", "")
        current_class = current_class.replace(".xlsx", "")
        current_class = current_class.replace("_language_mixed", "")
        current_class = current_class.title()
        # print(f)
        # imputation = 1 means there will be imputation, similar for stratify and classification. pass_value sets the grade
        # that is differential point between pass and fail.
        imputation = 0
        stratify = 0
        classification = 0
        pass_value = 5.5
        # percentage_full_row asks for the minimum % filled that each row needs to be allowed. percentage_full_column
        # is the same but then for columns
        percentage_full_row = 10  # %
        percentage_full_column = 10  # %
        final_data = pd.DataFrame()
        final_summised_scores = pd.DataFrame()
        k = 0
        for c in courses:
            # Looping through the different courses, each time changing the parameters based on the new course.
            course = ['nederlandse']
            course[0] = c
            current_course = courses_english[k]
            data = pd.concat([data_historie, data_actueel], axis=1)
            course_selector = data.filter(regex=c, axis=1)
            percentage_empty = course_selector.isna().sum() / data.shape[0] * 100
            percentage_empty = percentage_empty.sort_values()

            # Select the first course that adheres to the 65% full requirement
            for i in range(len(percentage_empty) - 1, 0, -1):
                if percentage_empty.iloc[i] < 35:
                    l = list(percentage_empty.index)
                    course[0] = l[i]
                    break
                elif i == 1:
                    l = list(percentage_empty.index)
                    course[0] = l[0]
            grade_number = course[0]
            grade_number = grade_number[-1]
            k = k + 1
            if int(grade_number) == 1:
                break
            q = max(int(grade_number) - 3, 1)
            for g in range(int(grade_number), q, -1):
                # Loop through each grade in a course.
                score = pd.DataFrame()
                summised_scores = pd.DataFrame()
                mod = 0
                # count through the models
                r = 0
                for m in models_regression:
                    data = 3
                    # only impute for Lasso and Ridge
                    if r < 2:
                        imputation = 0
                    else:
                        imputation = 1
                    r = r+1
                    X, y = get_x_and_y_sets(course, data, filename, imputation, classification, pass_value,
                                            percentage_full_row,
                                            percentage_full_column, g)
                    X_train, X_test, y_train, y_test = create_data_set(c, X, y, stratify)
                    chosen_model = m
                    summised_score = []
                    model, summised_score = create_model(chosen_model, X_train, X_test, y_train, y_test, X, y,
                                                         summised_score, multioutput)

                    if r > 2:
                        # Store the most important columns for Lasso and Ridge
                        arr = model.coef_
                        idx = (-arr).argsort()[:6]
                        idx = idx.tolist()
                        column_headers = X.columns.values.tolist()
                        most_influencial_courses_here = list(column_headers[i] for i in idx)
                        most_influencial_courses_here = [x[:-4] for x in most_influencial_courses_here]
                        most_influencial_courses = most_influencial_courses + most_influencial_courses_here

                    sc = []
                    mapie = MapieRegressor(estimator=model, cv='prefit', method='plus')
                    for i in alpha:
                        # run mapie for each significance level
                        mapie = mapie.fit(X, y.iloc[:, 0].ravel())
                        y_pred, y_pis = mapie.predict(X, alpha=i)

                        # Find the width of each prediction interval
                        width = 0
                        for j in range(X.shape[0]):
                            width = width + (y_pis[j, 1] - y_pis[j, 0])
                        width = width[0]
                        width = width / X.shape[0]
                        width = str(width)
                        width = width[:5]
                        sc.append(float(width))
                    score[models_regression_str[mod]] = sc
                    summised_scores[models_regression_str[mod]] = summised_score
                    mod = mod + 1
                # Every line after this is to plot and print all the received data
                plt.figure()
                for j in range(len(models_regression_str)):
                    plt.plot(alpha, score.iloc[:, j], 'o-', color=colours[j], alpha=0.8)

                plt.xlabel(r"Significance Level $\alpha$")
                plt.ylabel("Confidence Interval")
                plt.title(
                    f"{current_class} : {current_course} {g}"
                )
                plt.tight_layout()
                for spine in plt.gca().spines.values():
                    spine.set_visible(True)
                    spine.set_color('k')
                plt.xticks(alpha)
                plt.grid(visible=None)
                plt.legend(models_regression_str)
                plt.savefig(f'{filename_format} {current_class} for {c} {g}')
                # print("here")
                # plt.show()
                plt.close()
                final_data = pd.concat([final_data, score])
                final_summised_scores = pd.concat([final_summised_scores, summised_scores])
                # summised_scores.to_excel(f'{filename_format} {current_class} for {c} {g}.xlsx')
        if int(grade_number) == 1:
            break
        # print(final_data.groupby(level=0).mean())
        final_data = final_data.groupby(level=0).mean()
        plt.figure()
        for j in range(len(models_regression_str)):
            plt.plot(alpha, final_data.iloc[:, j], 'o-', color=colours[j], alpha=0.8)

        plt.xlabel(r"Significance Level $\alpha$")
        plt.ylabel("Confidence Interval")
        plt.title(
            f"{current_class} : average"
        )
        plt.tight_layout()
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('k')
        plt.xticks(alpha)
        plt.grid(visible=None)
        plt.legend(models_regression_str)

        plt.legend(models_regression_str)
        plt.savefig(f'{filename_format} {current_class} average scores')
        plt.close()
        final_summised_scores.to_excel(f'{filename_format} {current_class} all MAE.xlsx')
        final_summised_scores = final_summised_scores.groupby(level=0).mean()
        final_summised_scores.to_excel(f'{filename_format} {current_class} average.xlsx')
        # plt.show()
    freqs = Counter(most_influencial_courses)
    print(freqs.most_common(4))


def test_classification(directory, filename_format, courses):
    # This method is a culmination of many smaller methods, these were originally split but as time went on it proved (momentarily) easier to create this frankenstein.
    # This is the main thing that I would have done differently, I should have stuck to separate methods.
    print('classification')
    multioutput = 0

    course = ['nederlandse', 'engelse', 'wiskunde']
    for f in listdir(directory):
        # the next lines set up the parameters for each file
        # data = 0 means just historie, 1 = actueel, 2 is both, 3 is its own dataset
        data = 3
        filename = f'{filename_format}\\{f}'
        filename = f'{filename_format}\\{f}'
        current_class = f.replace("classes_actueel_", "")
        current_class = current_class.replace(".xlsx", "")
        current_class = current_class.replace("_language_mixed", "")
        current_class = current_class.title()
        print(f)
        # imputation = 1 means there will be imputation, similar for stratify and classification. pass_value sets the grade
        # that is differential point between pass and fail.
        imputation = 1
        stratify = 1
        classification = 1
        pass_value = 6
        # percentage_full_row asks for the minimum % filled that each row needs to be allowed. percentage_full_column
        # is the same but then for columns
        percentage_full_row = 10  # %
        percentage_full_column = 10  # %
        final_scores = pd.DataFrame()
        grade_number_list = ''
        for c in courses:
            # loop through the courses
            if multioutput == 0:
                course = ['nederlandse']
                course[0] = c
            final_data = pd.DataFrame()
            score = []
            data = pd.read_excel(filename)
            data.drop(columns=data.columns[0], axis=1, inplace=True)
            course_selector = data.filter(regex=c, axis=1)
            # find the course that has the minimum % students
            percentage_empty = course_selector.isna().sum() / data.shape[0] * 100
            percentage_empty = percentage_empty.sort_values()
            for i in range(len(percentage_empty) - 1, 0, -1):
                if percentage_empty.iloc[i] < 35:
                    l = list(percentage_empty.index)
                    course[0] = l[i]
                    break
                elif i == 1:
                    l = list(percentage_empty.index)
                    course[0] = l[0]
            grade_number = course[0]
            grade_number = grade_number[-1]
            q = max(int(grade_number) - 3, 1)
            for g in range(int(grade_number), q, -1):
                # loop through the grades in each course
                r = 0
                for m in models_classification:
                    # loop through the models
                    data = 3
                    if r < 2:
                        imputation = 0
                    else:
                        imputation = 1
                    r = r+1
                    X, y = get_x_and_y_sets(course, data, filename, imputation, classification, pass_value,
                                            percentage_full_row,
                                            percentage_full_column, g)
                    X_train, X_test, y_train, y_test = create_data_set(c, X, y, stratify)
                    chosen_model = m
                    summised_score = []
                    model, summised_score = create_model(chosen_model, X_train, X_test, y_train, y_test, X, y,
                                                         summised_score, multioutput)
                    y_pred = model.predict_proba(X_test)
                    if multioutput == 1:
                        y_pred = np.transpose([pred[:, 1] for pred in y_pred])
                        s = roc_auc_score(y_test, y_pred, average=None)
                        # print(s)
                    else:
                        try:
                            # Since some courses have only passes or fails they cannot get an AUC score
                            y_pred = y_pred[:, 1]
                            s = roc_auc_score(y_test.iloc[:, 0], y_pred, average=None)
                        except:
                            s = 0
                    score.append(s)
            # save, print and store all the found data
            if len(score) != 12:
                n = 12
                fill = [0] * n
                score = score[:n] + fill[len(score):]
            final_data[c] = score
            final_scores = pd.concat([final_scores, final_data])
            grade_number_list = grade_number_list + grade_number
            # print(f"For class {f}")
            # print(final_data)
            # final_data.to_csv(f'{f} classification data.csv', float_format="%2.2f")
        # print(final_scores.groupby(level=0).mean())
        print(f'{grade_number_list} for classification {filename_format} {current_class}')
        final_scores.to_excel(f'{filename_format} {current_class} classification data.xlsx', float_format="%2.3f")



colours = ['b', 'g', 'r', 'y']

# code below was build before the methods, as such it is repeated in each method.

course = ['nederlandse', 'engelse', 'wiskunde']
# course = []
multioutput = 0
# data = 0 means just historie, 1 = actueel, 2 is both, 3 is own dataset
data = 3
filename = 'Test sets\\classesfull_summised\\classes_actueel_all.xlsx'
# imputation = 1 means there will be imputation, similar for stratify and classification. pass_value sets the grade
# that is differential point between pass and fail.
imputation = 1
stratify = 0
classification = 1
pass_value = 6.0
# percentage_full_row asks for the minimum % filled that each row needs to be allowed. percentage_full_column
# is the same but then for columns
percentage_full_row = 10  # %
percentage_full_column = 10  # %
print("done")
pca = PCA()


# all models
# Don't need imputation:
# Pass fail/classification:
hgbm = HistGradientBoostingClassifier(random_state=42)
xgbc = XGBClassifier(objective='reg:squarederror')
# Exact grade:
xgbr = XGBRegressor(objective='reg:squarederror')
hgbr = HistGradientBoostingRegressor(random_state=42)

# need imputation:
# Pass fail/classification:
rfc = RandomForestClassifier(random_state=0)
blr = BaggingClassifier(estimator=LogisticRegression(max_iter=1000), n_estimators=10,
                        random_state=0)
ridgec = RidgeClassifier()
logr = LogisticRegression(random_state=0)

# Exact grade:
lr = LinearRegression()
lasso = linear_model.Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)

# create lists so we can loop through them
models_classification = [hgbm, xgbc, rfc, blr]
models_classification_str = ['hgbm', 'xgbc', 'rfc', 'bagging_logistic_regression']
models_regression = [hgbr, xgbr, lasso, ridge]
models_regression_str = ['hgbr', 'xgbr', 'lasso', 'ridge']

# leftover from before the methods.
chosen_model = rfc
summised_score = []

# sort all the groups of data together so we can loop through them.
# all groups together
summised = ['classesfull_summised_h_v', 'classesfull_summised', 'classesfull_summised_b']
summised_languages = ['classesfull_summised_languages_mixed', 'classesfull_summised_languages_mixed_b',
                      'classesfull_summised_languages_mixed_h_v']
courses_summised = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'wiskunde': ['wiskunde']}
courses_b = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'rekenen': ['rekenen 2f']}
courses_h_v = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'wiskunde': ['wiskunde a']}

courses_summised_list = [courses_h_v, courses_summised, courses_b]

# higher grades
bovenbouw = ['classesfull_bovenbouw']
courses_bovenbouw = {'nederlands': ['nederlandse taal en literatuur'], 'engels': ['engelse taal  en literatuur'],
                     'wiskunde': ['wiskunde a']}
courses_bovenbouw_list = [courses_bovenbouw]
# lower grades
onderbouw = ['classesfull_onderbouw', 'classesfull_onderbouw_b']
courses_onderbouw = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'wiskunde': ['wiskunde']}
courses_b = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'rekenen': ['rekenen 2f']}
courses_onderbouw_list = [courses_onderbouw, courses_b]

all_classes = ['classesfull_all']
courses_all = {'nederlands': ['nederlandse taal'], 'engels': ['engelse taal'], 'wiskunde': ['wiskunde']}

# group the groupings for easier looping.
courses_summery_regression = [courses_summised_list, courses_summised_list]
final_task_regression = [summised, summised_languages]

courses_summery_classification = [courses_summised_list,
                                  courses_summised_list]
final_task_classification = [summised, summised_languages]

# loop through all combinations of groups and levels.
m = 0
for a in final_task_regression:
    n = 0
    c = courses_summery_regression[m]
    for b in a:
        directory = fr'C:\Users\justu\PycharmProjects\pythonProject\Test sets\{b}'
        filename_format = f'Test sets\{b}'
        test_regression(directory, filename_format, c[n])
        n = n + 1
    m = m + 1

m = 0
for a in final_task_classification:
    n = 0
    c = courses_summery_classification[m]
    for b in a:
        directory = fr'C:\Users\justu\PycharmProjects\pythonProject\Test sets\{b}'
        filename_format = f'Test sets\{b}'
        test_classification(directory, filename_format, c[n])
        n = n + 1
    m = m + 1

# test
print("done")
