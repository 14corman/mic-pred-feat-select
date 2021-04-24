-- My own research

# Introduction
Currently, when a patient comes into a hospital with a bacterial infection, the patient will have the bacteria isolated, cultured, and colonies picked from the culture to be placed into multiple wells with growth media. Each well would have a unique concentration of an antibiotic. Multiple antibiotics can be tested together, but multiple antibiotics would not be placed into the same well. This process takes, on average, 72 hours to produce the [Minimum Inhibitory Concentration (MIC)](https://en.wikipedia.org/wiki/Minimum_inhibitory_concentration) value for the isolate (patient's isolated bacteria which could contain many bacterial cells) and a single antibiotic. That MIC value dictates how resistant or susceptible a bacterium is to a given antibiotic, and thus tells the physician whether a given antibiotic will be effective at clearing the bacteria from the patient. As time moves on, however, more resistant bacterial strains are becoming more prevalent, and more antibiotics are required to be tested before an adequate antibiotic may be found. There are many locations on the human body for bacterial infection, so each patient may be having a different outcome with their infection. However, with more resistant strains of bacteria, antibiotic takes longer to be given to that patient which makes the likelihood of death even more likely. This likelihood will continuously get higher until 2050 where it is predicted that bacterial infection will be the leading cause of death, replacing cancer , in the world (M. Bassetti). That is why we must make the time to produce an MIC for a patient's bacteria, antibiotic pair, smaller. Machine Learning, along with [Next Generation Sequencing (NGS)](https://www.illumina.com/science/technology/next-generation-sequencing.html), can take this time down to one to two hours for numerous antibiotics at once. NGS would be able to collect the raw read data within an hour, the data could be sent through a sequencing pipeline within a few minutes, and finally the output from the pipeline could be sent as input to a Machine Learning algorithm to have any number of antibiotic MIC predictions within a minute.

# Data
I plan on using annotated gene mutations (specifically Outer Membrane Protein (OMP) gene mutations) from Klebsiella Pneumoniae (K. pneu) bacteria to predict MIC values, given some antibiotics (actual antibiotics unknown at this time, have not chosen yet). Reasoning behind using K. pneu is due to it being a Gram-negative species, and its efficiency of gaining bacterial resistance. Gram-negative bacteria are more likely to gain antibiotic resistance, and most antibiotic resistant bacteria known are Gram-negative (Breijyeh, Z). The number of isolates that will be given as input is also unknown at this time. They have not been selected yet. Regarding antibiotic resistance, OMP genes may mutate to allow less antibiotic into the cell or expel more antibiotic once it enters the cell before the antibiotic has a chance to break the cell down. We will not be looking specifically for mutations that cause these changes. Instead, all mutations found for these genes will be given to the algorithms. It will be up to the algorithms to learn to find mutations that help predict the MIC value accordingly.

## Collecting data
These mutations will be collected from a pipeline that I built with JMI Laboratories which takes in a reference and raw read data, trims the read data, and does reference alignment between reference and trimmed read. Next, the pipeline finds [Single Nucleotide Polymorphisms (SNPs)](https://en.wikipedia.org/wiki/Single-nucleotide_polymorphism), insertions, deletions, and frameshifts and creates annotations from these mutations. The pipeline, then, creates an annotated Amino Acid (AA) sequence using the reference and annotations. Finally, it takes all isolates that were run, and the reference sequence, and performs a Multiple Sequence Alignment (MSA) so that all sequences match in length. This MSA is what the final output of the pipeline is for each gene reference used.

## Processing
The preprocessed data will have each AA be converted to an index for that species AA. For example, if we had a species that only has AAs A, B, and C, then their conversions would look like [A: 0, B: 1, C: 2]. That would mean an AA sequence "ACBCCA" would be converted into "021220"

# Algorithms
The three algorithms that will be tested are [Neural Networks (NN)](https://towardsdatascience.com/understanding-neural-networks-19020b758230), [Random Forests (RF)](https://towardsdatascience.com/understanding-random-forest-58381e0602d2), and [Gradient Boosted Forests (GBF)](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725). For GBF, I will be using an algorithm called XGBoost.


## Neural Networks
The authors in D. Aytan-Aktug, et. al. [3] found that a one-hidden-layer NN worked best when they did their study, and that the NN was prone to overfitting due to its complexity. However, other studies [9, 10] have shown other ways to generate NN models. In [9], a mix of multiple dense and drop-out layers were used while in [10] a Convolutional NN (CNN) was used. A CNN would not be advised here, as the authors in [9] made theirs due to having too many layers to start out. The authors in [9] only had four layers for each of the four nucleotides while here there are 25 different options that a position could be in a sequence (21 Amino Acids and four other characters). Because of the authors in D. Aytan-Aktug, et. al. [3] mentioning overfitting at one hidden layer, I will start with how they set up their model. If that performs poorly, then I will move to the model spoken about in [9].

# Training and Testing
When you train a Machine Learning algorithm, you obtain a model. A single algorithm can generate many different models depending on its input data, output prediction, the labels used to train, and parameters that make up the algorithm. The reasoning behind the inclusion labels can be thought of as follows, if you trained an algorithm to perform addition, for example, then gave it multiplication problems, it would fail. Even though both would have the same input and output, simple integers, the process to go from input to output has changed. That is why labels matter. Labels are used in training an algorithm. It tells the algorithm what is right after it tries to predict during training. The labels for addition may be 5 (for 3+2) and 6 (for 3+3). The labels for multiplication may be 6 (for 3\*2) and 9 (for 3\*3). The two different sets of labels would create two different models. One trained to predict addition and the other trained to predict multiplication. The same logic can be applied here. For the three algorithms mentioned previously, they will all be predicting MIC values. The labels, MICs, are different depending on antibiotic, so there will be a model trained for each antibiotic. If there would be three antibiotics, then there would be nine models total since we have three algorithms and three antibiotics.

In terms of splitting the dataset, 80% of the data will be made into the training dataset with the other 20% becoming the test dataset. When training a model, 5-fold Cross Validation will be performed. That means, out of the whole training dataset, 1/5 will be used to validate the model while the other 4/5 will be used to train the model.

# Hyperparameter Tuning
To give an algorithm the best chance at training a good model, you must give it proper hyperparameters. These hyperparameters are inputs given to the algorithms to guide the algorithm as it generates the model while training. Tuning these parameters can be done through a means called GridSearch [5] where you can think of the possible values for each hyperparameter coming together to form a grid and each point on the grid is a different model generated. In GridSearch, each model is generated in the grid and tested to give some performance value. Then, depending on the performance metric, you would maximize or minimize the models to get the best performaning one. This model would then have the proper hyperparameters you would want to use to train with. This can be seen as a brute force optimization method, and as the number of hyperparameters grow, and number of possible values per hyperparameter grow, the number of models needing to be trained increases exponentially. This is where $2^k$ factorial design [4] comes in. This approach makes it so that you only use two values per hyperparameter, a low value and a high value. The setup is the same as GridSearch, and we can still use the GridSearch functions, but instead of finding the exact, best hyperparameters, we only need to get a general idea on whether larger values or smaller values work best per parameter. From there, you can choose around the best value to use. For GBF and RF, this approach was used to tune multiple hyperparameters each. A 10-fold Cross Validation was used with GridSearch to test each model, and the performance metric used was the average F1 score (F1-micro score). GridSearch was also performed, as described above, for each antibiotic used. This would lead to a more generalized view for each hyperpameter as it looks at different labels (MICSs) from the different antibiotics. The results for these can be found in the `output\grid_search` folder as both plots and CSV output.

## XGBoost tuning
There were four parameters that were tuned with XGBoost; subsample (), max_depth (), learning_rate (), and colsample_bytree (). It was determined that only max_depth plays a role here as it was the only hyperparameter to make a difference between training with a low or high value. All other hyperparameters had similar results when comparing their low and high outputs. The max_depth hyperpameter had gave better performance with the smallest value rather than the default. This would mean that it is easy to overfit the data as having a smaller value forces the trees in the forest to only pick a tiny number of genes to use before stopping.

## RF tuning
There were three hyperparameters that were tuned for RF; max_depth (), min_samples_leaf (), min_samples_split (). Again, max_depth was the only hyperparameter whose value made a difference in the overall outcomce of the model. However, in this case it seems that the max_depth variable varied on whether a small value or large value worked best across the antibiotics. For three antibiotics, a larger max_depth was better, for one a small max_depth was better, and, for the last antibiotic, both large and small performed about the same. Because, in general, a larger max_depth worked best, that is what was chosen.

# Figures
The figures will be generated by predicting the test dataset.

## Figure 1
The figure I will try to reproduce is:

![Image1](figure_1.PNG)

Figure 1 is a [Receiver Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) plot. The plot tells how well an algorithm can predict for something when compared to chance (red dashed line). A perfect predictor would be one that has a line going up to (0.0, 1.0). In my case, I will either have three separate plots, one for each algorithm, or I will have one plot with all algorithms being compared. It will depend on the number of antibiotics being used which may lead to too many lines in the plot if all algorithms were combined into one.

## Figure 2
The other figure I will try to reproduce:

![Image2](figure_2.png)

Figure 2 is a symmetric jitter dot plot showing F1 scores for each Machine Learning algorithm over every antibiotic. [F1 scores](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) are another way to tell how well an algorithm can predict. Each point on the graph is an F1 score for a particular antibiotic/algorithm/MIC combination.

# Referenecs
1. M. Bassetti, G. Poulakou, E. Rupp´e, E. Bouza, S. J. V. Hal, and A. Brink,
“Antimicrobial resistance in the next 30 years, humankind, bugs and
drugs: a visionary approach.” Intensive Care Medicine, vol. 43, no. 10,
pp. 1464–1475, 2017.

2. Breijyeh, Z., Jubeh, B., & Karaman, R. (2020). Resistance of Gram-Negative Bacteria to Current Antibacterial Agents and Approaches to Resolve It. Molecules (Basel, Switzerland), 25(6), 1340. https://doi.org/10.3390/molecules25061340

3. D. Aytan-Aktug, P. T. L. C. Clausen, V. Bortolaia, F. M. Aarestrup, and O. Lund. "Prediction of Acquired Antimicrobial Resistance for Multiple Bacterial Species Using Neural Networks". American Society for Microbiology Journals, January 5, 2020, e00774-19. [https://doi.org/10.1128/MSYSTEMS.00774-19](https://doi.org/10.1128/MSYSTEMS.00774-19).

4. Winer, B. J., Brown, D. R. & Michels, K. M. Statistical principles in experimental design, vol. 2 (McGraw-Hill New York, 1971).

5. Lerman, P. M. Fitting Segmented Regression Models by Grid Search. Journal of the Royal Statistical Society. Series C (Applied
Statistics) 29, 77–84 (1980).

6. Neural Network (used in paper)

7. Random Forest

8. XGBoost

9. DeepArg

10. Predicting effects of noncoding variants with deep learning–based sequence model


## Data
[JMI Laboratories](https://www.jmilabs.com/)

345 Beaver Kreek Center, Suite A, North Liberty, IA 52317

# Reflection
One thing I need to note is collecting the data has taken longer than I initial thought it would take. A small sample of test data was collected, but I hope to get a large test sample to allow for a more diverse testing strategy, so replication becomes easier.

## Reflection update
I hope to have the data either 4/3 or 4/4. I have the data preprocessing code done. I just need to finish and test XGBoost once I have the data.

Cory Kromer-Edwards, B.A
University of Iowa
Computer Science PhD Student
Graduate Research Assistant
http://homepage.divms.uiowa.edu/~coryedwards/