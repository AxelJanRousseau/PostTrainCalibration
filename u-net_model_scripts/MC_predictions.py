import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from torch.utils.data import DataLoader

from config import dirs, settings
from my_utils import my_dataloader, get_model

select_train_folds = False
output_type = 'sigmoid'

nr_of_MC_samples = 20

batch_size = settings.U_NET_BATCHSIZE
pth = os.path.dirname(__file__)
data_path = dirs.DATA_PATH
model_path = dirs.MODELS_PATH
output_path = dirs.SAVED_SEG_MAPS
datasets = settings.DO_DATASET
model_weights = settings.WEIGHTS

dropout_decoders = False
dropout_center = True
dropout_rate = settings.MC_DROPOUT_RATE

dropout_settings = settings.DROPOUT_SETTINGS
from_learningrates = settings.MC_LR

if __name__ == '__main__':
    for dropout_setting in dropout_settings:
        for lr in from_learningrates:
            if dropout_setting == "center":
                name = "MC_center_lr_" + str(lr)
            elif dropout_setting == "decoder":
                name = "MC_decoders_lr_" + str(lr)
            for dataset_type in datasets:
                if dataset_type == 'BRATS_2018':
                    model = get_model.get_Brats_model(output_type, dropout_during_inference=True,
                                                      dropout_common_pathway=[dropout_rate],
                                                      dropout_setting=dropout_setting)
                    dataset_func = my_dataloader.BratsDataset
                if dataset_type == 'ISLES_2018':
                    model = get_model.get_Isles_18_model(output_type, dropout_during_inference=True,
                                                         dropout_common_pathway=[dropout_rate],
                                                         dropout_setting=dropout_setting)
                    dataset_func = my_dataloader.Isles18Dataset

                for weights_type in model_weights:

                    for i in range(5):  # folds
                        ds = dataset_func(os.path.join(data_path, dataset_type), i, train=select_train_folds,
                                          mask_type=output_type)
                        dataloader = DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False,
                                                collate_fn=my_dataloader.my_collate_fn)
                        model.load_weights(os.path.join(model_path, dataset_type, weights_type, name,
                                                        'Fold_{}_end.hdf5'.format(
                                                            i)))  # model_fold_{}.hdf5 Round_0_Fold_{}.hdf5
                        preds = []

                        for batch in dataloader:
                            in1, in2, gt = batch

                            predictions = np.zeros(gt.shape, dtype=np.float32)
                            for _ in range(nr_of_MC_samples):
                                predictions += model.predict([in1, in2]) / nr_of_MC_samples

                            if dataset_type == 'BRATS_2018':
                                predictions = predictions[:, 8:-8, 8:-8, 2:-2, :]
                            preds.append(predictions)
                        preds = np.vstack(preds)

                        pred_path = os.path.join(output_path, dataset_type, weights_type, name, "validation")

                        if not os.path.exists(pred_path):
                            os.makedirs(pred_path)

                        np.save(os.path.join(pred_path, "predictions_{}.npy".format(i)), preds)
