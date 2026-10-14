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

# Prediction: [34, 33, 33, 33]. Since 133 = 4 * 33 + 1, one fold gets one extra row.
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
print("Validation fold sizes:", [len(va_idx) for tr_idx, va_idx in cv.split(X_train, y_train)])

# Task 3:
# - cross_val_score for DummyClassifier(strategy="most_frequent") on X_train, y_train with that cv
# - Predict BEFORE running: roughly what mean accuracy, and why? (comment)
# - Print the scores and the mean

# Prediction: about 0.40. The majority class is 71 / 178 = 0.399 of the full data,
# and Day 1's baseline accuracy was 0.40.
baseline_scores = cross_val_score(DummyClassifier(strategy="most_frequent"), X_train, y_train, cv=cv)
print("Baseline CV:", baseline_scores)
print("Baseline mean:", baseline_scores.mean())

# Task 4:
# - cross_val_score for LogisticRegression(max_iter=5000), same data, same cv
# - Print the 4 scores, mean, and std
# - Comment: which fold is worst, and how many validation rows does ONE mistake cost in that fold (in points)?
# - Comment: does the model beat the baseline by more than the std? What does that tell you?

lr_scores = cross_val_score(LogisticRegression(max_iter=5000), X_train, y_train, cv=cv)
print("LR CV:", lr_scores)
print("LR mean:", lr_scores.mean())
print("LR std:", lr_scores.std())
# The third fold is worst: 0.9091 means 30 of 33 correct. One error costs
# 1 / 33 * 100 = 3.03 accuracy points in that fold.
# The mean gap is 0.9474 - 0.3984 = 54.9 points, versus 3.3 points of fold-to-fold
# standard deviation (about 17 times larger). This is not a significance test.

# Task 5 (the trap):
# - On the FULL X, y (not X_train): use KFold(n_splits=3) with NO shuffle
#   (from sklearn.model_selection import KFold)
# - Predict BEFORE running: class counts [n0, n1, n2] inside each of the 3 validation folds
#   (hint: print y.values[:5] and y.values[-5:], and think about the file order)
# - Print the real counts per fold with np.bincount(y.values[va_idx], minlength=3)
# - Repeat with KFold(n_splits=3, shuffle=True, random_state=1)
# - Comment: why is the first version dangerous for a model you want to trust?

# Prediction: without shuffling, the class-sorted rows should produce highly
# uneven folds. Shuffling should mix the classes, but does not guarantee balance.
no_shuffle_kf = KFold(n_splits=3)
for fold, (tr_idx, va_idx) in enumerate(no_shuffle_kf.split(X), start=1):
    print("No shuffle fold counts:", fold, np.bincount(y.values[va_idx], minlength=3))

shuffled_kf = KFold(n_splits=3, shuffle=True, random_state=1)
for fold, (tr_idx, va_idx) in enumerate(shuffled_kf.split(X), start=1):
    print("Shuffled fold counts:", fold, np.bincount(y.values[va_idx], minlength=3))

# Wine rows are stored sorted by class. Without shuffling, the counts are
# [59, 1, 0], [0, 59, 0], [0, 11, 48]; shuffled counts are
# [24, 23, 13], [17, 26, 16], [18, 22, 19]. Shuffling makes a mixed class
# distribution likely, not guaranteed; StratifiedKFold is what preserves
# class proportions across folds.
unshuffled_scores = cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=KFold(n_splits=3))
shuffled_scores = cross_val_score(
    LogisticRegression(max_iter=5000),
    X,
    y,
    cv=KFold(n_splits=3, shuffle=True, random_state=1),
)
print("Unshuffled KFold LR:", unshuffled_scores, "mean:", unshuffled_scores.mean())
print("Shuffled KFold LR:", shuffled_scores, "mean:", shuffled_scores.mean())
# The unshuffled mean of 0.316 is below the 0.398 baseline, while shuffling
# gives 0.950; the low score is a sign that this evaluation setup is broken.
# In fold 1, training has no class-0 wines ([0, 70, 48]) while validation has
# 59 class-0 wines. It gets only 1 of 60 right (about 0.017 accuracy).

# Task 6:
# - Fit LogisticRegression(max_iter=5000) on ALL of X_train, score on X_test ONCE
# - Comment: is that score inside "CV mean +/- std"? How many test rows is one mistake worth?

final_model = LogisticRegression(max_iter=5000)
final_model.fit(X_train, y_train)
print("Training accuracy:", final_model.score(X_train, y_train))
test_score = final_model.score(X_test, y_test)
print("Test accuracy:", test_score)
# Test accuracy is 0.9333, inside the CV range 0.9474 +/- 0.0329
# (0.9145 to 0.9803). One test error costs 1 / 45 * 100 = 2.22 points.

# Task 7 (one-line comment):
# - Why must you never use X_test while comparing models with cross-validation?

# Choosing a model by its test score optimistically biases that score because the winner was selected for doing well on those exact rows, leaving no honest final exam.

# Task 8:
# - On y_train (class counts 44 / 53 / 36), use StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
# - Predict BEFORE running: the ideal class counts per validation fold (do the divisions by hand, write them in a comment)
# - Print the real counts with np.bincount(y_train.values[va_idx], minlength=3)
#   (note: StratifiedKFold.split needs y: cv.split(X_train, y_train))
# - Comment: how close was your prediction? Why can't it be exact?

# Prediction by class (44 / 53 / 36 split into 5 folds):
# 44 / 5 = 8.8; 53 / 5 = 10.6; 36 / 5 = 7.2.
# class 0: [9, 9, 9, 9, 8]; class 1: [11, 11, 11, 10, 10];
# class 2: [8, 7, 7, 7, 7]. Each class's remainder means its counts cannot
# be identical in every fold.
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
for tr_idx, va_idx in cv5.split(X_train, y_train):
    print("Task 8 validation counts:", np.bincount(y_train.values[va_idx], minlength=3))
# The observed per-class counts match the predicted lists exactly; the extra
# row's fold could not be predicted because the remainders do not identify its
# destination.
