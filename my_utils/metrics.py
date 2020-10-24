import numpy as np

def binary_dice(y_true, y_pred, per_image=True, smooth=1e-7, norm='L1'):
    if norm == 'L1':
        intersection = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4) if per_image else None)
        union = np.sum(np.round(y_true), axis=(1, 2, 3, 4) if per_image else None) + np.sum(np.round(y_pred), axis=(1, 2, 3, 4) if per_image else None)
    elif norm == 'L2':
        intersection = np.sum(np.round(y_true) * np.round(y_pred), axis=(1, 2, 3, 4) if per_image else None)
        union = np.sum(np.round(y_true) ** 2, axis=(1, 2, 3, 4) if per_image else None) + np.sum(np.round(y_pred) ** 2, axis=(1, 2, 3, 4) if per_image else None)
    return (2 * intersection + smooth) / (union + smooth) if per_image else [(2 * intersection + smooth) / (union + smooth)]


def fast_ece(y_true, y_pred, n_bins = 10):
    ## ~sklearn code
    bins = np.linspace(0., 1.- 1./n_bins, n_bins) # alles >= laatste waarde word in extra bin gestoken, dus daarom deze rare notatie
    binids = np.digitize(y_pred, bins) - 1

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0 # don't use empty bins
    prob_true = (bin_true[nonzero] / bin_total[nonzero]) # acc
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero]) # conf

    weights = bin_total[nonzero] / np.sum(bin_total[nonzero])
    l1 = np.abs(prob_true-prob_pred)
    ece = np.sum(weights*l1)
    mce = l1.max()
    l1 = l1.sum()
    return {"acc": prob_true, "conf": prob_pred,"ECE": ece, "MCE": mce, "l1": l1}


def per_volume_masked_ECE(y_true, y_pred, masks, n_bins = 10):
    nr_volumes = len(y_true)
    result = np.zeros(nr_volumes)
    for i in range(nr_volumes):
        result[i]=fast_ece(y_true[i,masks[i]],y_pred[i,masks[i]],n_bins)["ECE"]
    return result



