import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.data import DataLoader

from config import dirs, settings
from my_utils import my_dataloader, get_model
from my_utils.lr_schedule import ReduceLROnPlateau, ModelCheckpointCb

Batch_Size = settings.U_NET_BATCHSIZE
learning_rate = settings.FINE_TUNE_LR
nr_folds = 5
nr_epochs = settings.FINE_TUNE_MAX_EPOCHS
data_path = dirs.DATA_PATH
model_path = dirs.MODELS_PATH
datasets = settings.DO_DATASET
model_weights = settings.WEIGHTS
output_type = 'sigmoid'
early_stopping_after = 6

plot_loss = False


def train_loop(model, dataLoader, valLoader, epochs, scheduler, checkpoint):
    validation_losses = []
    for epoch in range(epochs):

        model.fit(iter(dataLoader), verbose=2, sample_weight=None, class_weight=None)
        mean_train_loss = model.history.history['loss']
        # get evaluation at end of epoch

        mean_val_loss = model.evaluate(iter(valLoader), verbose=2)
        validation_losses.append(mean_val_loss)

        print(str(epoch) + " train loss: " + str(mean_train_loss) + "    validation loss: " + str(mean_val_loss))
        # save best model so far
        if checkpoint is not None:
            checkpoint.step(mean_val_loss, model)
            if checkpoint.epochs_no_improve >= 6:
                print("early_stopping")
                break
        if scheduler is not None:
            scheduler.on_epoch_end(epoch, mean_val_loss)
    return validation_losses


if __name__ == '__main__':
    for dataset_type in datasets:
        if dataset_type == 'BRATS_2018':
            model = get_model.get_Brats_model(output_type)
            dataset_func = my_dataloader.BratsDataset
        if dataset_type == 'ISLES_2018':
            model = get_model.get_Isles_18_model(output_type)
            dataset_func = my_dataloader.Isles18Dataset

        for weights_type in model_weights:
            val_loss_dict = {}
            for i in range(5):
                ds = dataset_func(os.path.join(data_path, dataset_type), i, train=True, mask_type=output_type)
                dataloader = DataLoader(ds, batch_size=Batch_Size, num_workers=2,
                                        collate_fn=my_dataloader.my_collate_fn_for_tf_iterator,
                                        shuffle=True)
                valds = dataset_func(os.path.join(data_path, dataset_type), i, train=False, mask_type=output_type)
                valdataloader = DataLoader(valds, batch_size=Batch_Size, num_workers=2,
                                           collate_fn=my_dataloader.my_collate_fn_for_tf_iterator,
                                           shuffle=False)
                model.load_weights(os.path.join(model_path, dataset_type, weights_type,
                                                'Fold_{}.hdf5'.format(i)))  # model_fold_{}.hdf5

                model.trainable = True
                assert (isinstance(model.layers[-4], tf.keras.layers.Conv3D))
                for layer in model.layers[:-4]:
                    layer.trainable = False
                adam = tf.keras.optimizers.Adam(lr=learning_rate)
                model.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy())

                save_path = os.path.join(model_path, dataset_type, weights_type, "fine_tune")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                checkpoint = ModelCheckpointCb(os.path.join(save_path, 'Fold_{}_checkpoint.hdf5'.format(i)))
                schedule = ReduceLROnPlateau(model=model, factor=0.1, patience=3, verbose=True)
                val_losses = train_loop(model, dataloader, valdataloader, epochs=nr_epochs, scheduler=schedule,
                                        checkpoint=checkpoint)
                val_loss_dict[i] = val_losses
                model.save_weights(os.path.join(save_path, 'Fold_{}_end.hdf5'.format(i)))
            if plot_loss:
                plt.figure()
                for fold in range(5):
                    plt.plot(val_loss_dict[fold], label=fold)
                plt.legend()
                plt.savefig("val_loss_finetune_{}_{}_startlr{}.png".format(dataset_type, weights_type.replace("/", "_"),
                                                                           learning_rate))
