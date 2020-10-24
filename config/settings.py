# select the dataset(s) to be used, BRATS_2018 and/or ISLES_2018
DO_DATASET = ["BRATS_2018", "ISLES_2018"]

# select weight(s). 'CE-SD','CE-CE','SD'
WEIGHTS = ['CE-SD','CE-CE','SD']

U_NET_BATCHSIZE = 2
AUX_BATCHSIZE = 64
AUX_KERNEL_SIZES = [1, 5]
AUX_LR = 5e-3
AUX_MAX_EPOCHS = 50

FINE_TUNE_LR = 1e-3
FINE_TUNE_MAX_EPOCHS = 50

MC_LR = [1e-4] #list to try multiple settings
MC_DROPOUT_RATE = 0.5
MC_MAX_EPOCHS = 50
DROPOUT_SETTINGS = ["center","decoder"]