import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from torch.utils.data import DataLoader

from config import dirs, settings
from my_utils import my_dataloader, get_model
from my_utils.cropping import extract_segment

select_train_folds = False
output_type = 'sigmoid'

batch_size = settings.U_NET_BATCHSIZE
data_path = dirs.DATA_PATH
model_path = dirs.MODELS_PATH
output_path = dirs.SAVED_SEG_MAPS
datasets = settings.DO_DATASET
model_weights = settings.WEIGHTS
use_checkpoint = False

if __name__ == '__main__':
    for dataset_type in datasets:
        if dataset_type == 'BRATS_2018':
            model = get_model.get_Brats_model(output_type)
            dataset_func = my_dataloader.BratsDataset
        if dataset_type == 'ISLES_2018':
            model = get_model.get_Isles_18_model(output_type)
            dataset_func = my_dataloader.Isles18Dataset
        for weights_type in model_weights:
            for i in range(5):
                ds = dataset_func(os.path.join(data_path, dataset_type), i, train=select_train_folds,
                                  mask_type=output_type)
                dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                        collate_fn=my_dataloader.my_collate_fn)

                file_name = 'Fold_{}_checkpoint.hdf5'.format(i) if use_checkpoint else 'Fold_{}_end.hdf5'.format(i)
                model.load_weights(os.path.join(model_path, dataset_type, weights_type, 'fine_tune',
                                                file_name))  # model_fold_{}.hdf5 Round_0_Fold_{}.hdf5
                preds = []
                for batch in dataloader:
                    in1, in2, gt = batch
                    out = model.predict([in1, in2])
                    if dataset_type == 'BRATS_2018':
                        # remove padding from output and ground truth.
                        # keep padding for isles for now, as GT shapes can be different per sample
                        out = [extract_segment(out[index], (120, 120, 78)) for index in range(out.shape[0])]
                        gt = [extract_segment(gt[index], (120, 120, 78)) for index in range(gt.shape[0])]
                    preds.append(out)

                preds = np.vstack(preds)

                if select_train_folds:
                    pred_path = os.path.join(output_path, dataset_type, weights_type, "fine_tune", "Training",
                                             "predictions_{}.npy".format(i))
                else:
                    pred_path = os.path.join(output_path, dataset_type, weights_type, "fine_tune", "validation",
                                             "predictions_{}.npy".format(i))

                # save outputs
                if not os.path.exists(os.path.dirname(pred_path)):
                    os.makedirs(os.path.dirname(pred_path))
                np.save(pred_path, preds)
