import numpy as np
# from pytorch_lightning.metrics import Metric
from torchmetrics import Metric
from ..dataloader.pc_dataset import get_SemKITTI_label_name


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


class IoU(Metric):
    def __init__(self, dataset_config, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.hist_list = []
        self.best_miou = 0
        self.SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
        self.unique_label = np.asarray(sorted(list(self.SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [self.SemKITTI_label_name[x] for x in self.unique_label + 1]

    def update(self, predict_labels, val_pt_labs) -> None:
        # print(f"Predicted: {predict_labels}")
        # print(f"Labels: {val_pt_labs}")
        self.hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, self.unique_label))

    def compute(self):
        #####
        # During sanity check
        #   - First two runs get working hist_list(s)
        #   - The last run gets an empty hist_list which leads to an error in the per_class_iu function:
        #           ValueError: Input must be 1- or 2-d.

        # Crashes in VALIDATION/TEST EPOCH END STEP
        #####
        iou = per_class_iu(sum(self.hist_list))
        if np.nanmean(iou) > self.best_miou:
            self.best_miou = np.nanmean(iou)
        # self.hist_list = [] #### EDIT
        return iou, self.best_miou