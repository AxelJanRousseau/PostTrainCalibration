import numpy as np
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from config import dirs, settings
from tqdm import tqdm
import matplotlib.pyplot as plt

Batch_Size = settings.AUX_BATCHSIZE
learning_rate = settings.AUX_LR

nr_folds = 5
nr_epochs = settings.AUX_MAX_EPOCHS

pth = os.path.dirname(__file__)
data_path = dirs.SAVED_SEG_MAPS
model_path = dirs.MODELS_PATH

train_on_weights = settings.WEIGHTS
kernel_sizes = settings.AUX_KERNEL_SIZES
datasets = settings.DO_DATASET
device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')

early_stopping_after =6
plot_loss=False

class ModelCheckpointCb:
    def __init__(self) -> None:
        super().__init__()
        self.best_loss = float('Inf')
        self.epochs_no_improve = 0
        self.save_file = None
        self.threshold = 1e-4

    def step(self, loss, model):
        if loss < self.best_loss * ( 1 - self.threshold ):
            print("saving best model")
            self.best_loss = loss
            self.epochs_no_improve = 0
            self.save_file=model.state_dict()
        else:
            self.epochs_no_improve += 1


def train_loop(model, dataLoader, valLoader, epochs, optimizer, loss_function, scheduler, checkpoint):
    model.to(device)

    validation_losses = []
    for epoch in range(epochs):

        model.train()
        train_batch_losses = []
        for x, y in tqdm(dataLoader):
            x = x.to(device)
            y = y.squeeze().type(torch.float32).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.item())
        # get evaluation at end of epoch
        model.eval()
        validate_batch_losses = []
        with torch.no_grad():
            for x, y in valLoader:
                x = x.to(device)
                y = y.squeeze().type(torch.float32).to(device)
                out = model(x)
                loss = loss_function(out.squeeze(), y)
                validate_batch_losses.append(loss.item())
        mean_val_loss = np.mean(validate_batch_losses)
        mean_train_loss = np.mean(train_batch_losses)
        validation_losses.append(mean_val_loss)
        print(str(epoch) + " train loss: " + str(mean_train_loss) + "    validation loss: " + str(mean_val_loss))
        # save best model so far
        if checkpoint is not None:
            checkpoint.step(mean_val_loss, model)
            if checkpoint.epochs_no_improve >= early_stopping_after:
                print("early_stopping")
                break
        # lower lr on plateau
        if scheduler is not None:
            scheduler.step(mean_val_loss)
    return model, validation_losses


def get_data_loader(path,gt_path, fold, shuffle, training):
    # turn 3d volumes into slices and return dataloader
    if training:
        preds = np.load(os.path.join(path,"Training", "predictions_{}.npy".format(fold)))
        gt = np.load(os.path.join(gt_path,"Training", "GT_{}.npy".format(fold)))
    else:
        preds = np.load(os.path.join(path,"validation", "predictions_{}.npy".format(fold)))
        gt = np.load(os.path.join(gt_path,'validation', "GT_{}.npy".format(fold)))
    preds = torch.tensor(preds).permute(0, 3, 4, 1, 2)
    gt = torch.tensor(gt).permute(0, 3, 4, 1, 2)
    preds = preds.reshape(preds.shape[0] * preds.shape[1], *preds.shape[2:])
    gt = gt.reshape(gt.shape[0] * gt.shape[1], *gt.shape[2:])

    dataset = torch.utils.data.TensorDataset(preds, gt)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, shuffle=shuffle, num_workers=2,
                                             pin_memory=True)
    return dataLoader


if __name__ == '__main__':

    for dataset in datasets:
        for train_on in train_on_weights:
            dataset_path = os.path.join(data_path,dataset,train_on,"base_model")
            gt_path = os.path.join(data_path,dataset,"GT")
            for k_size in kernel_sizes:
                val_loss_dict = {}
                for fold in range(nr_folds):

                    dataLoader = get_data_loader(dataset_path,gt_path, fold=fold, shuffle=True,training=True)
                    valLoader = get_data_loader(dataset_path,gt_path, fold=fold, shuffle=False,training=False)

                    c, h, w = 1, dataLoader.dataset[0][0].shape[-1], dataLoader.dataset[0][0].shape[-1]
                    auxiliary = torch.nn.Conv2d(1, 1, k_size, padding=k_size // 2, bias=True)



                    optimizer = optim.Adam(auxiliary.parameters(), lr=learning_rate, weight_decay=0.)
                    schedule = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
                    loss_function = BCEWithLogitsLoss()

                    checkpointer = ModelCheckpointCb()


                    model, val_loss=train_loop(auxiliary, dataLoader, valLoader, nr_epochs, optimizer, loss_function, schedule, checkpointer)
                    val_loss_dict[fold]=val_loss

                    save_path = os.path.join(model_path,dataset,train_on,'auxiliary')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    torch.save(checkpointer.save_file,os.path.join(save_path,'fold_{}_k_{}_filter_checkpoint.pth'.format(fold, k_size)))
                    torch.save(model.state_dict(),os.path.join(save_path,'fold_{}_k_{}_filter_end.pth'.format(fold,k_size)))
                if plot_loss:
                    plt.figure()
                    for fold in range(nr_folds):
                        plt.plot(val_loss_dict[fold], label=fold)
                    plt.legend()

                    plt.savefig("val_loss_{}_{}_k{}_startlr{}.png".format(dataset,train_on.replace("/","_"),k_size,learning_rate))