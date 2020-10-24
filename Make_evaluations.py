import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from my_utils import metrics
import torch
from sklearn.metrics import brier_score_loss, jaccard_score, f1_score
from config import dirs, settings
from tensorflow import sigmoid
import os
import pandas as pd
from scipy.stats import ttest_rel,wilcoxon

data_path = dirs.SAVED_SEG_MAPS
result_path = dirs.RESULT_FILE_PATH
nr_folds = 5
nr_bins = 20

datasets = settings.DO_DATASET

methods = ["base_model",'fine_tune',"auxiliary"]
trained_weights = settings.WEIGHTS
kernel_sizes = settings.AUX_KERNEL_SIZES

if __name__ == '__main__':
    for dropout_setting in settings.DROPOUT_SETTINGS:
        if dropout_setting == "center":
            name = "MC_center_lr_"
        elif dropout_setting == "decoder":
            name = "MC_decoders_lr_"
        all_names = [name+str(lr) for lr in settings.MC_LR]
        methods.extend(all_names)

    for dataset in datasets:

        for weights in trained_weights:
            result_list = [] # group results per dataset and weights
            for method in methods:
                for k in kernel_sizes:
                    mean_dice = np.array([])
                    per_volume_ece = np.array([])
                    mean_ece = 0.
                    mean_masked_ece = 0.
                    mean_mce = 0.
                    mean_l1 = 0.
                    mean_l1_error = 0.

                    print(data_path, dataset, weights, method)
                    all_preds = []
                    all_gt = []
                    all_masks = []

                    for fold in range(nr_folds):

                        if 'auxiliary' in method:
                            filename = "predictions_{}_k_{}.npy".format(fold,k)
                        else:
                            filename = "predictions_{}.npy".format(fold)
                        try:
                            preds = np.load(os.path.join(data_path, dataset, weights, method, "validation", filename))
                            gt = np.load(os.path.join(data_path, dataset, "GT", "validation", "GT_{}.npy".format(fold)))
                            masks = np.load(os.path.join(data_path, dataset,"masks", "validation","mask_{}.npy".format(fold)))
                        except Exception as e:
                            print(e)
                            # continue
                        if preds.shape[-1] !=1 and gt.shape[-1] ==1:
                            preds = preds[...,None]
                        assert(gt.shape == masks.shape)

                        p_max = preds.max()
                        if p_max > 1.1:
                            # preds are logits
                            preds = sigmoid(preds).numpy()
                        all_preds.append(preds)
                        all_gt.append(gt)
                        all_masks.append(masks)
                    all_preds = np.vstack(all_preds)
                    all_gt = np.vstack(all_gt)
                    all_masks = np.vstack(all_masks)

                    mean_dice = metrics.binary_dice(all_gt,all_preds)

                    per_volume_ece =metrics.per_volume_masked_ECE(all_gt, all_preds,all_masks, n_bins=nr_bins)


                    if method == "base_model":
                        base_results = [mean_dice,per_volume_ece]
                        p_dice = 0
                        p_ece = 0
                    elif len(mean_dice) != len(base_results[0]):
                        p_dice = 0
                        p_ece = 0
                    else:
                        p_dice = wilcoxon(base_results[0],mean_dice).pvalue
                        p_ece = wilcoxon(base_results[1], per_volume_ece).pvalue

                    mean_dice = mean_dice.mean()
                    per_volume_ece = per_volume_ece.mean()



                    result_list.append([method, k, mean_dice,p_dice,per_volume_ece*100,p_ece])
                    if not 'auxiliary' in method:
                        break
            df = pd.DataFrame(result_list, columns=['method', "k",'dice', 'p_val_dice','per_volume_ece%','pval_ece'])
            df = df.round({'dice':3,'p_val_dice':4,'per_volume_ece%':4,'pval_ece':4})
            df.set_index(['method', "k"])
            path = os.path.join(result_path, dataset, weights)
            if not os.path.exists(path):
                os.makedirs(path)
            df.to_csv(os.path.join(path, 'results.csv'))
