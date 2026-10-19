# Datasets: toy points (Task 0-1), then load_diabetes (as_frame=True, return_X_y=True)
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
A = np.array([[1], [2], [3], [4]])
b = np.array([[1], [3], [2], [4]])

# Task 0 (by hand, comments only):
# - Toy data: x = [1, 2, 3, 4], y = [1, 3, 2, 4]
# - Compute by hand: mean x, mean y, slope (use the formula from the example), intercept
# - Compute by hand: the 4 predictions, the 4 errors, MAE, and the lazy guesser's MAE (always predicts mean y)
# - Compute by hand: R^2 (squared errors of your line vs of the lazy guesser)
# - Predict: what will the line predict for x = 5?

# mean of x = 2.5, mean of y = 2.5
# slope = Σ(x − 2.5)(y − 2.5) ÷ Σ(x − 2.5)²
# Top: (−1.5)(−1.5) + (−0.5)(0.5) + (0.5)(-0.5) + (1.5)(1.5) = 4. Bottom: 2.25 + 0.25 + 0.25 + 2.25 = 5. So slope = 0.8.
# intercept = mean y − slope × mean x = 2.5 − 0.8 × 2.5 = 0.5
# Line: prediction = 0.8 × x + 0.5.

# Line predictions: 1.3, 2.1, 2.9, 3.7.
# Line errors (true - predicted): -0.3, 0.9, -0.9, 0.3.
# Line MAE = 0.6; baseline predictions are all 2.5, so baseline MAE = 1.0.
# Line SSE = 1.8; baseline SSE = 5. Therefore baseline R^2 = 0 and line R^2 = 1 - 1.8 / 5 = 0.64.
# The line predicts 4.5 for x = 5.

# Task 1:
# - Fit LinearRegression on the same toy data in code (X must be 2-D: np.array([[1],[2],[3],[4]]))
# - Print slope (coef_), intercept_, predictions, MAE, R^2, and model.predict([[5]])
# - Comment: do all of them match your hand answers? If not, find your mistake

toy_model = LinearRegression().fit(A, b)
pred = toy_model.predict(A)
print(f"slope: {toy_model.coef_[0]}")
print(f"intercept: {toy_model.intercept_}")
print(f"predictions: {pred}")
print(f"MAE: {mean_absolute_error(b, pred)}")
print(f"R^2: {toy_model.score(A, b)}")
print(f"Prediction for x=5: {toy_model.predict([[5]])}")

# Yes all match. The slope and intercept are the same, the predictions are the same, the MAE is the same, and the R^2 is the same. The prediction for x=5 is also the same.

# Task 2:
# - Load diabetes. Split with test_size=0.2, random_state=3 (NO stratify)
# - Predict BEFORE running: len(X_train) and len(X_test) for 442 rows (use the ceil rule from Day 1)
# - Keep the test set out of fitting and tuning; use it only for the evaluations requested in Tasks 3-5

X, y = load_diabetes(as_frame=True, return_X_y=True)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(len(X_train), len(X_test))  # 353  89

# Task 3:
# - DummyRegressor(strategy="mean") fit on the TRAIN data
# - Predict BEFORE running: what will its R^2 on the test set be, roughly, and why?
# - Print MAE, RMSE, R^2 on the test set, and set(base.predict(X_test)) rounded
# - Comment: what single number does it predict for every row, and where does it come from?

base = DummyRegressor(strategy="mean").fit(X_train, y_train)
# Prediction: test R^2 should be near zero and could be negative because this model ignores the features and predicts the training-target mean. Its prediction is the mean of y_train.
pred = base.predict(X_test)
baseline_mae = mean_absolute_error(y_test, pred)
print(f"Baseline MAE: {baseline_mae}")
print(f"Baseline RMSE: {root_mean_squared_error(y_test, pred)}")
print(f"Baseline R^2: {base.score(X_test, y_test)}")
print(f"Baseline prediction (all rows, rounded): {set(np.round(pred, 2))}")

# Task 4:
# - LinearRegression on ALL 10 features: fit on train, print MAE, RMSE, R^2 on test (once)
# - Comment: by what percent is its MAE lower than the baseline's? Is train R^2 much higher than test R^2?
# - Print model.coef_.shape. Comment: why 10? Which coefficient has the largest absolute value?
# - Comment: why would it be a mistake to say "that feature is the most important one"?

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
model_mae = mean_absolute_error(y_test, pred)
full_model_test_r2 = r2_score(y_test, pred)
print(f"MAE: {model_mae}")
print(f"RMSE: {root_mean_squared_error(y_test, pred)}")
print(f"Test R^2: {full_model_test_r2}")
print(f"Train R^2: {model.score(X_train, y_train)}")
print(f"MAE reduction from baseline: {(baseline_mae - model_mae) / baseline_mae * 100:.1f}%")
print(f"Model coefficients shape: {model.coef_.shape}")
largest_coefficient_index = np.abs(model.coef_).argmax()
print(f"Largest absolute coefficient: {X.columns[largest_coefficient_index]} ({model.coef_[largest_coefficient_index]:.3f})")
# Coefficient magnitude alone does not establish feature importance: correlated features can share or distort their apparent effects, and coefficient size depends on feature scale.


# Task 5:
# - Fit a second LinearRegression using ONLY the "bmi" column (X_train[["bmi"]], double brackets)
# - Predict BEFORE running: will its test R^2 be higher or lower than the 10-feature model? Why?
# - Print its R^2 and MAE. Comment: what did dropping 9 features cost you, in R^2 points?

second_model = LinearRegression().fit(X_train[["bmi"]], y_train)
# Prediction: I expect the one-feature model to have lower test R^2 because it cannot use the other predictors; test R^2 is not guaranteed to decrease when features are removed.
pred = second_model.predict(X_test[["bmi"]])
single_feature_test_r2 = r2_score(y_test, pred)
print(f"Second model R^2: {single_feature_test_r2}")
print(f"Second model MAE: {mean_absolute_error(y_test, pred)}")
print(f"Test R^2 change after dropping 9 features: {single_feature_test_r2 - full_model_test_r2:.3f}")

# Task 6:
# - cv = KFold(n_splits=5, shuffle=True, random_state=3) on X_train, y_train
# - Predict BEFORE running: the 5 fold sizes (353 rows)
# - Run cross_val_score with scoring="neg_mean_absolute_error" for LinearRegression AND for DummyRegressor
# - Predict BEFORE running: will the raw numbers be positive or negative, and why?
# - Print the raw scores, then positive MAE mean and std for both models
# - Comment: does the model beat the baseline by more than the std? By how many MAE points?

cv = KFold(n_splits=5, shuffle=True, random_state=3)
# Prediction: the fold sizes will be [71, 71, 71, 70, 70]. The scores will be negative because scikit-learn negates MAE so that larger scores remain better.
lr_scores = cross_val_score(LinearRegression(), X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
baseline_scores = cross_val_score(DummyRegressor(strategy="mean"), X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
print(f"LR CV scores: {lr_scores}")
print(f"Baseline CV scores: {baseline_scores}")
lr_mae_mean = -lr_scores.mean()
baseline_mae_mean = -baseline_scores.mean()
baseline_mae_std = baseline_scores.std()
mae_improvement = baseline_mae_mean - lr_mae_mean
print(f"LR MAE mean: {lr_mae_mean:.3f}, std: {lr_scores.std():.3f}")
print(f"Baseline MAE mean: {baseline_mae_mean:.3f}, std: {baseline_mae_std:.3f}")
print(f"Mean MAE improvement: {mae_improvement:.3f}")
print(f"Improvement exceeds baseline fold-to-fold std: {mae_improvement > baseline_mae_std}")
# This compares the mean improvement with the baseline's fold-to-fold standard deviation; it is descriptive, not a statistical significance test.

# Task 7 (the traps, two small ones):
# - Try train_test_split(X, y, test_size=0.2, random_state=3, stratify=y) inside try/except and print the error type
# - Predict BEFORE running: will it work? Comment: why can't you stratify a continuous number?
# - Try mean_squared_error(y_test, pred, squared=False) inside try/except. Print sklearn.__version__ and the error
# - Comment: what is the correct modern replacement?

print(f"Sklearn version: {sklearn.__version__}")
try:
    X_strat_train, X_strat_test, y_strat_train, y_strat_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
    print("Stratified split worked.")
except ValueError as error:
    print(f"Error type: {type(error).__name__}: {error}")

try:
    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f"RMSE: {rmse}")
except TypeError as error:
    print(f"Error type: {type(error).__name__}: {error}")
# Prediction: stratification should fail because this target is continuous, while stratification requires discrete class labels. The modern RMSE function is root_mean_squared_error(y_true, y_pred), used above.