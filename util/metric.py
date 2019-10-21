"""
Metrics for computing evalutation results
"""

import numpy as np

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_runs:
            number of test runs
    """
    def __init__(self, max_label=20, n_runs=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_runs = 1 if n_runs is None else n_runs

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]

    def record(self, pred, target, labels=None, n_run=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            if target_idx_j:  # if ground-truth contains this class
                tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)

    def get_mIoU(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mIoU_binary(self, n_run=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
                      for run in range(self.n_runs)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU
