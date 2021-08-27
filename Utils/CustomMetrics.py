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


"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid) 
reid/evaluation_metrics/ranking.py. Modifications: 
1) Only accepts numpy data input, no torch is involved.
1) Here results of each query can be returned.
2) In the single-gallery-shot evaluation case, the time of repeats is changed 
   from 10 to 100.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
  mask = np.zeros(num, dtype=np.bool)
  for _, indices in ids_dict.items():
    i = np.random.choice(indices)
    mask[i] = True
  return mask


def cmc(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    topk=100,
    separate_camera_set=False,
    single_gallery_shot=False,
    first_match_break=False,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape
  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1)
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute CMC for each query
  ret = np.zeros([m, topk])
  is_valid_query = np.zeros(m)
  num_valid_queries = 0
  for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    if separate_camera_set:
      # Filter out samples from same camera
      valid &= (gallery_cams[indices[i]] != query_cams[i])
    if not np.any(matches[i, valid]): continue
    is_valid_query[i] = 1
    if single_gallery_shot:
      repeat = 100
      gids = gallery_ids[indices[i][valid]]
      inds = np.where(valid)[0]
      ids_dict = defaultdict(list)
      for j, x in zip(inds, gids):
        ids_dict[x].append(j)
    else:
      repeat = 1
    for _ in range(repeat):
      if single_gallery_shot:
        # Randomly choose one instance for each id
        sampled = (valid & _unique_sample(ids_dict, len(valid)))
        index = np.nonzero(matches[i, sampled])[0]
      else:
        index = np.nonzero(matches[i, valid])[0]
      delta = 1. / (len(index) * repeat)
      for j, k in enumerate(index):
        if k - j >= topk: break
        if first_match_break:
          ret[i, k - j] += 1
          break
        ret[i, k - j] += delta
    num_valid_queries += 1
  if num_valid_queries == 0:
    raise RuntimeError("No valid query")
  ret = ret.cumsum(axis=1)
  if average:
    return np.sum(ret, axis=0) / num_valid_queries
  return ret, is_valid_query


def mean_ap(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """

  # -------------------------------------------------------------------------
  # The behavior of method `sklearn.average_precision` has changed since version
  # 0.19.
  # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
  # (https://github.com/zhunzhong07/person-re-ranking/
  # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
  # (http://www.liangzheng.org/Project/project_reid.html).
  # My current awkward solution is sticking to this older version.

  # -------------------------------------------------------------------------

  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape

  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1)
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute AP for each query
  aps = np.zeros(m)
  is_valid_query = np.zeros(m)
  for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    y_true = matches[i, valid]
    y_score = -distmat[i][indices[i]][valid]
    if not np.any(y_true): continue
    is_valid_query[i] = 1
    aps[i] = average_precision_score_market(y_true, y_score)
  if len(aps) == 0:
    raise RuntimeError("No valid query")
  if average:
    return float(np.sum(aps)) / np.sum(is_valid_query)
  return aps, is_valid_query

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