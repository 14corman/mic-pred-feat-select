# MPGA
Mic Prediction using Gene Annotation

# Introduction

In this project, I will be using my research. Right now, when a patient comes into a hospital with a bacterial infection, it takes, on average, 72 hours to produce the Minimum Inhibitory Concentration (MIC) value for that patient's bacteria and an antibiotic. That MIC value dictates how resistant or susceptible a bacterium is to a given antibiotic, and thus tells the physician whether a given antibiotic will be effective at clearing the bacteria from the patient. This 72-hour window is extremely large though, and patients have a high likelihood of not surviving. Machine Learning, along with Next Generation Sequencing (NGS), can take this down to a few hours. From this, I plan on using annotated gene mutations from bacteria to predict MIC values, given some antibiotic. The input gene annotations will be in two forms:

1. Human Genome Variation Society (HGVS) nomenclature (EX: R52\_D53insAC)
2. Annotated Amino Acid (AA) sequences of genes (EX: ...RACD...)

This information is like how D. Aytan-Aktug et al. viewed their input data (converting HGVS annotations into points using BLOSUM62 matrix and marking AAs in sequences with either 1 for mutation detected and -1 for no mutation detected), but I will be predicting MIC values rather than gene mutations that cause resistance.

# Figures

## Figure 1
The figure I will try to reproduce is:

![Image1](figure_1.PNG)

Figure 1 is a [Receiver Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) plot. The plot tells how well an algorithm can predict for something when compared to chance (red dashed line). A perfect predictor would be one that has a line going up to (0.0, 1.0). In my case, I will still have 2 plots like what is shown, but my 2 plots will be for the Random Forest and Neural Network.

## Figure 2
The other figure I will try to reproduce:

![Image2](figure_2.png)

Figure 2 is a symmetric jitter dot plot showing F1 scores for each Machine Learning algorithm over every antibiotic. [F1 scores](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) are another way to tell how well an algorithm can predict. Each point on the graph is an F1 score for a particular antibiotic/algorithm/MIC combination.

# Materials and methods

## Sequences
The gene sequences are collected and sequenced using Illumina sequencing at JMI Laboratories. The sequences are then run through an annotation pipeline that I have made to generate the annotations and annotated AA sequences.

## Preprocessing
### HGVS annotations
Convert annotation into a score using BLOSUM62 substitution matrix. Inserts and deletions are counted as 1 positon regardless of how many AA were inserted or delted. This would essentially be a list of scores for each bacterial isolate.

### AA sequences
Each position in the sequence is marked with:
* 1 - if a mutation occurred at that position in the sequence (inserts and deletions count as a single position)
* -1 - no mutation occurred at that position in the sequence

## Algorithms
There are two algorithms that will be used for prediction:
1. [Neural Network](https://towardsdatascience.com/understanding-neural-networks-19020b758230)
2. [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)

## Steps
1. Collect input data from JMI and Annotation pipeline
2. Preprocess that data as described above
3. Separate processed dataset into Training and Testing dataset
4. Train Neural Network and Random Forest separately on same Training dataset
5. Test both algorithms with the Testing dataset (capture predictions from this)
6. Create ROC plot and F1 scores from predictions and actual MICs used in previous step
7. Create final ROC plot (figure 1) and F1 scores plot (figure 2)

# References

## Paper
D. Aytan-Aktug, P. T. L. C. Clausen, V. Bortolaia, F. M. Aarestrup, and O. Lund. "Prediction of Acquired Antimicrobial Resistance for Multiple Bacterial Species Using Neural Networks". American Society for Microbiology Journals, January 5, 2020, e00774-19. [https://doi.org/10.1128/MSYSTEMS.00774-19](https://doi.org/10.1128/MSYSTEMS.00774-19).

## Data
[JMI Laboratories](https://www.jmilabs.com/)

345 Beaver Kreek Center, Suite A, North Liberty, IA 52317