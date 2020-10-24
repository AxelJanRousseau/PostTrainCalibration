import os

import numpy as np
from torch.utils.data import DataLoader

from config import dirs, settings
from my_utils import my_dataloader, get_model
from my_utils.cropping import extract_segment

is_training_list = [False, True]
output_type = 'linear'  # select logit output

batch_size = settings.U_NET_BATCHSIZE
data_path = dirs.DATA_PATH
model_path = dirs.MODELS_PATH
output_path = dirs.SAVED_SEG_MAPS
datasets = settings.DO_DATASET  # BRATS_2018 and/or ISLES_2018
model_weights = settings.WEIGHTS

if __name__ == '__main__':
    for dataset_type in datasets:
        if dataset_type == 'BRATS_2018':
            model = get_model.get_Brats_model(output_type)
            dataset_func = my_dataloader.BratsDataset
        if dataset_type == 'ISLES_2018':
            model = get_model.get_Isles_18_model(output_type)
            dataset_func = my_dataloader.Isles18Dataset
        for weights_type in model_weights:
            for is_training in is_training_list:
                # Dice_score = 0.
                for i in range(5):
                    ds = dataset_func(os.path.join(data_path, dataset_type), i, train=is_training,
                                      mask_type=output_type)
                    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                            collate_fn=my_dataloader.my_collate_fn)
                    model.load_weights(os.path.join(model_path, dataset_type, weights_type, 'Fold_{}.hdf5'.format(i)))
                    preds = []
                    gts = []
                    masks = []
                    for batch in dataloader:
                        in1, in2, gt = batch

                        out = model.predict([in1, in2])
                        if dataset_type == 'BRATS_2018':
                            # remove padding from output and ground truth.
                            # keep padding for isles for now, as GT shapes can be different per sample
                            out = [extract_segment(out[index], (120, 120, 78)) for index in range(out.shape[0])]
                            gt = [extract_segment(gt[index], (120, 120, 78)) for index in range(gt.shape[0])]
                            in2 = [extract_segment(in2[index], (120, 120, 78)).astype('bool') for index in
                                    range(in2.shape[0])]
                        preds.append(out)
                        gts.append(gt)
                        masks.append(in2.astype('bool'))
                    preds = np.vstack(preds)
                    gts = np.vstack(gts)
                    masks = np.vstack(masks)

                    if is_training:
                        pred_path = os.path.join(output_path, dataset_type, weights_type, "base_model", "Training",
                                                 "predictions_{}.npy".format(i))
                        gt_path = os.path.join(output_path, dataset_type, "GT", "Training", "GT_{}.npy".format(i))
                        mask_path = os.path.join(output_path, dataset_type, "masks", "Training",
                                                 "mask_{}.npy".format(i))
                    else:
                        pred_path = os.path.join(output_path, dataset_type, weights_type, "base_model", "validation",
                                                 "predictions_{}.npy".format(i))
                        gt_path = os.path.join(output_path, dataset_type, "GT", "validation", "GT_{}.npy".format(i))
                        mask_path = os.path.join(output_path, dataset_type, "masks", "validation",
                                                 "mask_{}.npy".format(i))

                    # save outputs
                    if not os.path.exists(os.path.dirname(pred_path)):
                        os.makedirs(os.path.dirname(pred_path))
                    np.save(pred_path, preds)
                    if not os.path.exists(os.path.dirname(gt_path)):
                        os.makedirs(os.path.dirname(gt_path))
                    np.save(gt_path, gts)
                    if not os.path.exists(os.path.dirname(mask_path)):
                        os.makedirs(os.path.dirname(mask_path))
                    np.save(mask_path, masks)
