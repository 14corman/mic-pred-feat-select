- My own research

# Introduction
Currently, when a patient comes into a hospital with a bacterial infection, the patient will have the bacteria isolated, cultured, and colonies picked from the culture to be placed into multiple wells with growth media. Each well would have a unique concentration of an antibiotic. Multiple antibiotics can be tested together, but multiple antibiotics would not be placed into the same well. This process takes, on average, 72 hours to produce the [Minimum Inhibitory Concentration (MIC)](https://en.wikipedia.org/wiki/Minimum_inhibitory_concentration) value for the isolate (patient's isolated bacteria which could contain many bacterial cells) and a single antibiotic. That MIC value dictates how resistant or susceptible a bacterium is to a given antibiotic, and thus tells the physician whether a given antibiotic will be effective at clearing the bacteria from the patient. As time moves on, however, more resistant bacterial strains are becoming more prevalent, and more antibiotics are required to be tested before an adequate antibiotic may be found. There are many locations on the human body for bacterial infection, so each patient may be having a different outcome with their infection. However, with more resistant strains of bacteria, antibiotic takes longer to be given to that patient which makes the likelihood of death even more likely. This likelihood will continuously get higher until 2050 where it is predicted that bacterial infection will be the leading cause of death, replacing cancer , in the world (M. Bassetti). That is why we must make the time to produce an MIC for a patient's bacteria, antibiotic pair, smaller. Machine Learning, along with [Next Generation Sequencing (NGS)](https://www.illumina.com/science/technology/next-generation-sequencing.html), can take this time down to one to two hours for numerous antibiotics at once. NGS would be able to collect the raw read data within an hour, the data could be sent through a sequencing pipeline within a few minutes, and finally the output from the pipeline could be sent as input to a Machine Learning algorithm to have any number of antibiotic MIC predictions within a minute.

# Data
I plan on using annotated gene mutations (specifically Outer Membrane Protein (OMP) gene mutations) from Klebsiella Pneumoniae (K. pneu) bacteria to predict MIC values, given some antibiotics (actual antibiotics unknown at this time, have not chosen yet). Reasoning behind using K. pneu is due to it being a Gram-negative species, and its efficiency of gaining bacterial resistance. Gram-negative bacteria are more likely to gain antibiotic resistance, and most antibiotic resistant bacteria known are Gram-negative (Breijyeh, Z). The number of isolates that will be given as input is also unknown at this time. They have not been selected yet. Regarding antibiotic resistance, OMP genes may mutate to allow less antibiotic into the cell or expel more antibiotic once it enters the cell before the antibiotic has a chance to break the cell down. We will not be looking specifically for mutations that cause these changes. Instead, all mutations found for these genes will be given to the algorithms. It will be up to the algorithms to learn to find mutations that help predict the MIC value accordingly. 

## Collecting data
These mutations will be collected from a pipeline that I built with JMI Laboratories which takes in a reference and raw read data, trims the read data, and does reference alignment between reference and trimmed read. Next, the pipeline finds [Single Nucleotide Polymorphisms (SNPs)](https://en.wikipedia.org/wiki/Single-nucleotide_polymorphism), insertions, deletions, and frameshifts and creates annotations from these mutations. Finally the pipeline creates an annotated Amino Acid (AA) sequence using the reference and annotations.

## Forms of data and processing
 The algorithm input will be in three forms of gene annotations. The first will be a list of annotations given in [Human Genome Variation Society (HGVS)](https://www.hgvs.org/) nomenclature (EX: R52\_D53insAC). The second form of input will be the Annotated AA sequences of genes (EX: ...RACD...). The paper I will be getting the figure from (D. Aytan-Aktug) used similar input and found that a combination of the two forms performed best, so I will be doing that as well. The authors in that paper used a [Blosum62](https://en.wikipedia.org/wiki/BLOSUM) substitution matrix to convert the HGVS annotations into substitution scores, and the AA sequences were converted so that each position in the sequence was either a 1 (a SNP, insertion, or deletion was found) or -1 (no mutation occurred). Insertions and deletions were given a single 1 regardless of size. The third form is like the second in that it will be a list of AAs. However, the third input form will have each AA be converted to an index for that species AA. For example, if we had a species that only has AAs A, B, and C, then their conversions would look like [A: 0, B: 1, C: 2]. That would mean an AA sequence "ACBCCA" would be converted into "021220". I will also try a combination of the first form and last form as input.

 With that said, there will be two main types of input. Each type will be a combination of the forms mentioned above. The first type will be the combination of forms one and two. The second type of input will be the combination of forms one and three. So, you can think that each type will have the HGVS annotation list, and be looking at the same annotated AA sequence. The main difference between the two types is how the annotated AA sequence is processed.

# Algorithms
The three algorithms that will be tested are [Neural Networks (NN)](https://towardsdatascience.com/understanding-neural-networks-19020b758230), [Random Forests (RF)](https://towardsdatascience.com/understanding-random-forest-58381e0602d2), and [Gradient Boosted Forests (GBF)](https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725). The authors in D. Aytan-Aktug, et. al. found that a one-hidden-layer NN worked best when they did their study, and that the NN was prone to overfitting due to its complexity. Because of this, I will also start out with one-hidden-layer. If underfitting is found, I will add another one or two layers. Otherwise, the one-hidden-layer model will be taken as best performant NN model. 

# Training and Testing
When you train a Machine Learning algorithm, you obtain a model. A single algorithm can generate many different models depending on its input data, output prediction, the labels used to train, and parameters that make up the algorithm. The reasoning behind the inclusion labels can be thought of as follows, if you trained an algorithm to perform addition, for example, then gave it multiplication problems, it would fail. Even though both would have the same input and output, simple integers, the process to go from input to output had changed. That is why labels matter. Labels are used in training an algorithm. It tells the algorithm what is right after it tries to predict during training. The labels for addition may be 5 (for 3+2) and 6 (for 3+3). The labels for multiplication may be 6 (for 3\*2) and 9 (for 3\*3). The two different sets of labels would create two different models. One trained to predict addition and the other trained to predict multiplication. The same logic can be applied here. For the three algorithms mentioned previously, they will all be predicting MIC values. They may either have input data being combination of forms one and two or a combination of forms one and three. The labels though are different depending on antibiotic. So, there will be two models per antibiotic. One model for input forms one and two, and one model for input forms one and three. If there would be four antibiotics, then there would be eight models trained per Machine Learning algorithm. In terms of splitting the dataset, 80% of the data will be made into the training dataset with the other 20% becoming the test dataset. When training a model, 5-fold Cross Validation will be performed. That means, out of the whole training dataset, 1/5 will be used to validate the model while the other 4/5 will be used to train the model.

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
M. Bassetti, G. Poulakou, E. Rupp´e, E. Bouza, S. J. V. Hal, and A. Brink,
“Antimicrobial resistance in the next 30 years, humankind, bugs and
drugs: a visionary approach.” Intensive Care Medicine, vol. 43, no. 10,
pp. 1464–1475, 2017.

Breijyeh, Z., Jubeh, B., & Karaman, R. (2020). Resistance of Gram-Negative Bacteria to Current Antibacterial Agents and Approaches to Resolve It. Molecules (Basel, Switzerland), 25(6), 1340. https://doi.org/10.3390/molecules25061340

D. Aytan-Aktug, P. T. L. C. Clausen, V. Bortolaia, F. M. Aarestrup, and O. Lund. "Prediction of Acquired Antimicrobial Resistance for Multiple Bacterial Species Using Neural Networks". American Society for Microbiology Journals, January 5, 2020, e00774-19. [https://doi.org/10.1128/MSYSTEMS.00774-19](https://doi.org/10.1128/MSYSTEMS.00774-19).


## Data
[JMI Laboratories](https://www.jmilabs.com/)

345 Beaver Kreek Center, Suite A, North Liberty, IA 52317

# Reflection
One thing I need to note is collecting the data has taken longer than I initial thought it would take. A small sample of test data was collected, but I hope to get a large test sample to allow for a more diverse testing strategy so replication becomes easier.