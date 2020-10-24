import numpy as np
import torch
import os
from config import dirs, settings


nr_folds = 5
Batch_Size = 128

seg_path = dirs.SAVED_SEG_MAPS  # validation , Training
model_path = dirs.MODELS_PATH
weights = settings.WEIGHTS # 'Checkpoints\CE-SD',"Checkpoints\CE-CE"
method_name = 'auxiliary' # gebruik enkel covolutional, bilateral ect werken toch niet
kernel_sizes = settings.AUX_KERNEL_SIZES  #[1, 5, 7, 9, 11]


datasets = settings.DO_DATASET
device = 'cuda'
use_checkpoint = False


def get_dataLoader(ds_path,fold):
    
    preds = np.load(os.path.join(ds_path,"validation", "predictions_{}.npy".format(fold)))
    original_shape = preds.shape
    preds = torch.tensor(preds).permute(0, 3, 4, 1, 2)
    preds = preds.reshape(preds.shape[0] * preds.shape[1], *preds.shape[2:])
    dataset = torch.utils.data.TensorDataset(preds)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)
    return dataLoader,original_shape


if __name__ == '__main__':
    for dataset in datasets:
        for weight_type in weights:
            dataset_path = os.path.join(seg_path, dataset, weight_type, "base_model")
            # gt_path = os.path.join(data_path,dataset,"GT")

            for fold in range(nr_folds):
                dataLoader, original_shape = get_dataLoader(dataset_path,fold)
                nr,x_pos,y_pos,z_pos,c_num = original_shape
                for k_size in kernel_sizes:
                    weight_path = os.path.join(model_path, dataset, weight_type, method_name)
                    model_file_name= "fold_{}_k_{}_filter_checkpoint.pth".format(fold, k_size) if use_checkpoint else "fold_{}_k_{}_filter_end.pth".format(fold, k_size)
                    model = torch.nn.Conv2d(1, 1, k_size, padding=k_size // 2, bias=True)
                    model.load_state_dict(torch.load(os.path.join(weight_path, model_file_name)))
                    model = torch.nn.Sequential(model, torch.nn.Sigmoid())
                    model.to(device)
                    model.eval()
                    preds = []
                    with torch.no_grad():
                        for x in dataLoader:
                            x = x[0].to(device)
                            preds.append(model(x).detach().cpu().numpy())
                    preds = np.vstack(preds).squeeze()

                    preds = preds.reshape(nr,z_pos,x_pos,y_pos)
                    preds = preds.transpose(0,2,3,1)
                    pred_path = os.path.join(seg_path, dataset, weight_type, method_name, "validation", "predictions_{}_k_{}.npy".format(fold, k_size))
                    if not os.path.exists(os.path.dirname(pred_path)):
                        os.makedirs(os.path.dirname(pred_path))

                    np.save(pred_path,preds)


