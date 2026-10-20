# Datasets: toy points (Task 0-1), then load_diabetes (as_frame=True, return_X_y=True)
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
X_toy = np.array([[1], [2], [3], [4]])
y_toy = np.array([1, 3, 2, 4])

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

toy_model = LinearRegression().fit(X_toy, y_toy)
toy_pred = toy_model.predict(X_toy)
print(f"slope: {toy_model.coef_[0]:.1f}")
print(f"intercept: {toy_model.intercept_:.1f}")
print(f"predictions: {toy_pred}")
print(f"MAE: {mean_absolute_error(y_toy, toy_pred):.2f}")
print(f"R^2: {toy_model.score(X_toy, y_toy):.2f}")
print(f"Prediction for x=5: {toy_model.predict([[5]])}")

# Yes all match. The slope and intercept are the same slope = 0.8, intercept = 0.5, the predictions are the same, the MAE is the same MAE = 0.60, and the R^2 is the same R^2 = 0.64. The prediction for x=5 is also the same 4.5 at x =5.

# Task 2:
# - Load diabetes. Split with test_size=0.2, random_state=3 (NO stratify)
# - Predict BEFORE running: len(X_train) and len(X_test) for 442 rows (use the ceil rule from Day 1)
# - Keep the test set out of fitting and tuning; use it only for the evaluations requested in Tasks 3-5

X, y = load_diabetes(as_frame=True, return_X_y=True)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(len(X_train), len(X_test))  # 442 × 0.2 = 88.4, rounded up to 89 test rows so 353 train rows.

# Task 3:
# - DummyRegressor(strategy="mean") fit on the TRAIN data
# - Predict BEFORE running: what will its R^2 on the test set be, roughly, and why?
# - Print MAE, RMSE, R^2 on the test set, and set(base.predict(X_test)) rounded
# - Comment: what single number does it predict for every row, and where does it come from?

base = DummyRegressor(strategy="mean").fit(X_train, y_train)
# Prediction: test R^2 should be near zero and could be negative because this model ignores the features and predicts the training-target mean. Its prediction is the mean of y_train which is 153.56.
base_pred = base.predict(X_test)
baseline_mae = mean_absolute_error(y_test, base_pred)
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline RMSE: {root_mean_squared_error(y_test, base_pred):.2f}")
print(f"Baseline R^2: {base.score(X_test, y_test):.2f}")
print(f"Baseline prediction (all rows, rounded): {set(np.round(base_pred, 2))}")

# Task 4:
# - LinearRegression on ALL 10 features: fit on train, print MAE, RMSE, R^2 on test (once)
# - Comment: by what percent is its MAE lower than the baseline's? Is train R^2 much higher than test R^2?
# - Print model.coef_.shape. Comment: why 10? Which coefficient has the largest absolute value?
# - Comment: why would it be a mistake to say "that feature is the most important one"?

model = LinearRegression().fit(X_train, y_train)
lr_pred = model.predict(X_test)
model_mae = mean_absolute_error(y_test, lr_pred)
full_model_test_r2 = r2_score(y_test, lr_pred)
print(f"MAE: {model_mae:.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, lr_pred):.2f}")
print(f"Test R^2: {full_model_test_r2:.2f}")
print(f"Train R^2: {model.score(X_train, y_train):.2f}")
print(f"MAE reduction from baseline: {(baseline_mae - model_mae) / baseline_mae * 100:.1f}%")
print(f"Model coefficients shape: {model.coef_.shape}")
largest_coefficient_index = np.abs(model.coef_).argmax()
print(f"Largest absolute coefficient: {X.columns[largest_coefficient_index]} ({model.coef_[largest_coefficient_index]:.3f})")
# Train R^2 (0.5366) is 0.1204 higher than test R^2 (0.4162), suggesting some overfitting and weaker performance on unseen data.
# There are 10 coefficients because the model learns one slope for each of the 10 input features.
# The largest absolute coefficient belongs to s1(-938.24), but that alone does not make it the most important feature.
# In this dataset, s1 and s2 are strongly correlated (r = 0.90), so their coefficients can share explanatory signal and be difficult to interpret separately.
# The features are on comparable normalized scales, so scale differences are not the main caveat here.

# Task 5:
# - Fit a second LinearRegression using ONLY the "bmi" column (X_train[["bmi"]], double brackets)
# - Predict BEFORE running: will its test R^2 be higher or lower than the 10-feature model? Why?
# - Print its R^2 and MAE. Comment: what did dropping 9 features cost you, in R^2 points?

second_model = LinearRegression().fit(X_train[["bmi"]], y_train)
# Prediction: I expect the one-feature model to have lower test R^2 because it cannot use the other predictors; test R^2 is not guaranteed to decrease when features are removed.
bmi_pred = second_model.predict(X_test[["bmi"]])
single_feature_test_r2 = r2_score(y_test, bmi_pred)
print(f"Second model R^2: {single_feature_test_r2:.2f}")
print(f"Second model MAE: {mean_absolute_error(y_test, bmi_pred):.2f}")
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
# The model's mean MAE is 22.33 points lower than the baseline's. This exceeds the model and baseline fold-to-fold standard deviations of 3.75 and 3.29 points, respectively. This is descriptive, not a statistical significance test.

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
    rmse = mean_squared_error(y_test, bmi_pred)
    print(f"RMSE: {rmse:.2f}")
except TypeError as error:
    print(f"Error type: {type(error).__name__}: {error}")
# Prediction: stratification should fail because this target is continuous, while stratification requires discrete class labels. The modern RMSE function is root_mean_squared_error(y_true, y_pred), used above.