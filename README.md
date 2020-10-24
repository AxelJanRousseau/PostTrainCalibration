# PostTrainCalibration

This repository contains source code for "Post training uncertainty calibration of deep networks for medical image segmentation"

## instructions:
1. Obtain model weights and pre-process data as in https://github.com/JeroenBertels/optimizingdice
2. Place model weights in the appropriate subfolders in ./Models. rename the files to Fold_0.hdf5, Fold_1.hdf5,...
3. Place cases of the datasets in the ./Datasets/BRATS_2018 and ./Datasets/ISLES_2018 folders
4. run the retrain_dropout.py and fintune_base_model.py scripts to train the dropout and fine-tune methods
5. run make_base_model_predictions.py and then Train_aux.py to train the auxiliary models.
6. save_aux_segmaps.py, MC_predictions.py and make_fine_tune_predictions.py to save the segmentation outputs
7. Finally, run Make_evaluations.py to calculate Dice and ECE scores.
