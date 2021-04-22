# Grid Search
This folder contains all Grid Search output files. The files are boken up by algorithm used. So, for example, all GridSearch results for Random Forests have file names startting with `rf_`.

## CSV
There are 2 kinds of CSV files. One that has results separated by antibiotic and one that has all results put together. The files that just have a name containing the algorithm name are the combined CSV files. These files have a column deticated to showing the antibiotic used. They also have the metadata removed that the antibiotic files contain.

## Plot files
The plot files are also given. These show the low and high values used in the $2^k$ factorial design approach to GridSearch. Each point is a model trained and tested. Since all hyperparameters were given together, each model in the plots was trained with different values for the other hyperparemeters being tuned. 

## Runs
Some of the files for XGBoost have an extra part to the name, `run_#`. This is used to separate the different runs of GridSearch depending on what was being tuned. `run_1` was tuning all 4 hyperparameters for XGBoost, `run_2` was tuning both learning rate and max_depth together, and no run (technically `run_3`) is just tuning max_depth.