import numpy as np


# ref: https://github.com/VisualComputingInstitute/triplet-reid/blob/master/evaluate.py
def average_precision_score_market(y_true, y_score):
    """ Compute average precision (AP) from prediction scores.
    This is a replacement for the scikit-learn version which, while likely more
    correct does not follow the same protocol as used in the default Market-1501
    evaluation that first introduced this score to the person ReID field.
    Args:
        y_true (array): The binary labels for all data points.
        y_score (array): The predicted scores for each samples for all data
            points.
    Raises:
        ValueError if the length of the labels and scores do not match.
    Returns:
        A float representing the average precision given the predictions.
    """

    if len(y_true) != len(y_score):
        raise ValueError('The length of the labels and predictions must match '
                         'got lengths y_true:{} and y_score:{}'.format(
                            len(y_true), len(y_score)))

    # Mergesort is used since it is a stable sorting algorithm. This is
    # important to compute consistent and correct scores.
    y_true_sorted = y_true[np.argsort(y_score)]

    tp = np.cumsum(y_true_sorted)
    total_true = np.sum(y_true_sorted)
    recall = tp / total_true
    recall = np.insert(recall, 0, 0.)
    precision = tp / np.arange(1, len(tp) + 1)
    precision = np.insert(precision, 0, 1.)
    ap = np.sum(np.diff(recall) * ((precision[1:] + precision[:-1]) / 2))

    return ap


y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

aps = []

for i in range(len(y_true)):
    aps.append(average_precision_score_market(y_true[i], y_pred[i]))
print(np.average(aps))
