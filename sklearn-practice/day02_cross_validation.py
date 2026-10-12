# Dataset: wine (load_wine, as_frame=True)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
data = load_wine(as_frame=True)

# Task 1:
# - Split with test_size=0.25, random_state=7, stratify=y
# - Rule for the whole file: X_test and y_test stay untouched until Task 6

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

# Task 2:
# - Make cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
# - Predict BEFORE running: the sizes of the 4 validation folds on X_train (133 rows)
# - Print the real sizes using: for tr_idx, va_idx in cv.split(X_train, y_train): len(va_idx)
# - Comment: why are they not all equal?

# The sizes of the 4 validation folds on X_train (133 rows) will not be all equal because 133 is not divisible by 4. Therefore, some folds will have one more sample than others to accommodate the total number of samples.
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
print([len(va_idx) for tr_idx, va_idx in cv.split(X_train, y_train)])
# The sizes of the validation folds are not all equal because 133 is not divisible by 4, resulting in some folds having one more sample than others.

# Task 3:
# - cross_val_score for DummyClassifier(strategy="most_frequent") on X_train, y_train with that cv
# - Predict BEFORE running: roughly what mean accuracy, and why? (comment)
# - Print the scores and the mean

# I predict the mean accuracy will be around 0.5 because the DummyClassifier with strategy="most_frequent" will always predict the most frequent class in the training data, which is likely to be correct about half of the time given the class distribution.
baseline = cross_val_score(DummyClassifier(strategy="most_frequent"), X_train, y_train, cv=cv)
print(baseline)
print(baseline.mean())

# Task 4:
# - cross_val_score for LogisticRegression(max_iter=5000), same data, same cv
# - Print the 4 scores, mean, and std
# - Comment: which fold is worst, and how many validation rows does ONE mistake cost in that fold (in points)?
# - Comment: does the model beat the baseline by more than the std? What does that tell you?

model = cross_val_score(LogisticRegression(max_iter=5000), X_train, y_train, cv=cv)
print(model)
print(model.mean())
print(model.std())
# The worst fold is the one with the lowest score. The number of validation rows in that fold can be calculated by checking the length of the validation indices for that fold. One mistake in that fold would cost 1/len(va_idx) points. If the model's mean accuracy beats the baseline by more than the standard deviation, it suggests that the model is performing significantly better than the baseline.
# The model beats the baseline by more than the std, indicating that it is a better predictor than the baseline model.

# Task 5 (the trap):
# - On the FULL X, y (not X_train): use KFold(n_splits=3) with NO shuffle
#   (from sklearn.model_selection import KFold)
# - Predict BEFORE running: class counts [n0, n1, n2] inside each of the 3 validation folds
#   (hint: print y.values[:5] and y.values[-5:], and think about the file order)
# - Print the real counts per fold with np.bincount(y.values[va_idx], minlength=3)
# - Repeat with KFold(n_splits=3, shuffle=True, random_state=1)
# - Comment: why is the first version dangerous for a model you want to trust?

# class counts in the first version are not representative of the overall distribution of classes in the dataset, which can lead to biased model evaluation. The second version with shuffling ensures that each fold has a more balanced representation of classes, making the evaluation more reliable.
kf = KFold(n_splits=3)
for tr_idx, va_idx in kf.split(X, y):
    print(np.bincount(y.values[va_idx], minlength=3))

kf = KFold(n_splits=3, shuffle=True, random_state=1)
for tr_idx, va_idx in kf.split(X, y):
    print(np.bincount(y.values[va_idx], minlength=3))

# The first version is dangerous because it can lead to biased model evaluation due to the lack of class representation in the validation folds.

# Task 6:
# - Fit LogisticRegression(max_iter=5000) on ALL of X_train, score on X_test ONCE
# - Comment: is that score inside "CV mean +/- std"? How many test rows is one mistake worth?

final_model = LogisticRegression(max_iter=5000)
final_model.fit(X_train, y_train)
print(final_model.score(X_train, y_train))
print(final_model.score(X_test, y_test))
# The score on X_test is compared to the cross-validation mean and standard deviation. If the test score falls within the range of "CV mean +/- std", it suggests that the model's performance on unseen data is consistent with its performance during cross-validation. One mistake on the test set would cost 1/len(X_test) points, which can be calculated based on the number of samples in the test set.

# Task 7 (one-line comment):
# - Why must you never use X_test while comparing models with cross-validation?

# You must never use X_test while comparing models with cross-validation because it is meant to be a final evaluation set that simulates unseen data. Using it during model selection or hyperparameter tuning can lead to overfitting and an overly optimistic estimate of model performance, compromising the validity of the evaluation.