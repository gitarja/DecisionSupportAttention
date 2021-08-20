"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""

import numpy as np


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    results = []
    for q_idx in range(num_q):
        tmp = {}
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        results.append(tmp)
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    # TODO: extract retrieval results in here
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


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
    y_true_sorted = y_true[np.argsort(-y_score, kind='mergesort')]

    tp = np.cumsum(y_true_sorted)
    total_true = np.sum(y_true_sorted)
    recall = tp / total_true
    recall = np.insert(recall, 0, 0.)
    precision = tp / np.arange(1, len(tp) + 1)
    precision = np.insert(precision, 0, 1.)
    ap = np.sum(np.diff(recall) * ((precision[1:] + precision[:-1]) / 2))

    return ap