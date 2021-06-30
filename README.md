# MPGA
Mic Prediction using Gene Annotation

### More info
More information can be found under the manuscript directory within the README file.

# Materials and methods

## Sequences
3 Outer Membrane Protein (OMP) genes (OMPK35, OMPK36, OMPK37) were collected from Klebsila Pneumoniae (KPN).

## Algorithms
There are two algorithms that will be used for prediction:
1. [Neural Network](https://towardsdatascience.com/understanding-neural-networks-19020b758230)
2. [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
3. [Gradient Boosted Forests](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725) (More specifically, [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html))

## Steps
1. Collect input data from JMI and Annotation pipeline
2. Preprocess that data
3. Separate processed dataset into Training and Testing dataset
4. Train Neural Network, Random Forest, and XGBoost separately on same Training dataset
5. Test algorithms with the Testing dataset (capture predictions from this)
