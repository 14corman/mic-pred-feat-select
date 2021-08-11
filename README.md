# MIC prediction and feature selection
MIC Prediction and FeatureSelection using XGBoost for Neural Networks

## Data
contig fasta files were collected from Klebsila Pneumoniae (KPN).

## Algorithms
There are two algorithms that will be used for prediction:
1. [Neural Network](https://towardsdatascience.com/understanding-neural-networks-19020b758230)
3. [Gradient Boosted Forests](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725) (More specifically, [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html))

## Steps
Note that all python run commands below must be done within the `script` directory.
1. Collect Fastas and MIC CSV file (put them in `data` folder)
2. Run `python get_format_data.py`
3. Zip up `train.libsvm` and put that zip file in an S3 bucket
4. Put `xgboost.ipynb` in Sagemaker, modify to point to `train.zip` in S3, and run all cells
5. Collect trained XGBoost model and control XGBoost model and put them in `output/models`
6. Run `python xgboost_test.py` to get `feature_importance.csv` and test results for XGBoost models
7. Use `feature_importance.csv` to modify `nn.py` with most important features (all features with importance >=10 in sorted order)
8. Run `python nn.py files` to generate NN training, validation, and testing files. Take note of the console output to update the `nn.py` file.
9. Run `python nn.py cv` to do grid search using 10-fold cross validation to determine best learning rate, architecture combination. Update `BEST_LEARNING_RATE`, `BEST_ARCH_SIZE`, and `BEST_DROPOUT_PROB` at top of nn.py when finished.
10. Run `python nn.py train` and note the file name of the best model hdf5 file. Modify `nn.py` with that name (delete all other hdf5 files)
11. Run `python nn.py test` to test NN model
12. Run `python nn_analysis.py` to generate error rate CSV and MIC heatmap.

For more information/detail on each step, visit the respective folders/script files.

## Testing your own data
If you would like to test your own data with our best NN model, follow these steps. You should be in the repository's main directory to call step 2.
1. Collect Fastas and MIC CSV file (put them in `data` folder)
2. Run `python script` to run test pipeline

Because testing the NN model does not require changing any of the files, the base file for `script` was made into the test pipeline. This pipeline will run the same things that the following 4 lines would run, but automatically.
1. Run `python get_format_data.py test`
2. Run `python nn.py files` to generate NN test file.
3. Run `python nn.py test` to test NN model
4. Run `python nn_analysis.py` to generate error rate CSV and MIC heatmap.
