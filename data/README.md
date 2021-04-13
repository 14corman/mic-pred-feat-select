# Input Data
All data will be held here. There will be 3 places data will be held:
1. main folder (hidden): this is where the actual data will be held when running the code, but cannot be put on Github
2. test folder: A set of files for seeing how real data will be provided, and to build and test code with. Anyone wanting to replicate the findings will be able to use this test folder to test.
3. processed folder (may not be made/hidden): The processed folder will be the place where processed data is stored. It may not be created initially as it will be made programatically.

## Main folder
Since the `main` folder is not included in this repo, I will say a bit about what you should expect to find in there. A `main` folder is required in this directory, and that is where all original data will be stored. The program will look at this folder to collect data to be processed. There should be the following files in that folder:
1. antibiotics_OMPK35.tsv
2. antibiotics_OMPK36.tsv
3. antibiotics_OMPK37.tsv
4. export_msa_75_OMPK35.csv
5. export_msa_75_OMPK36.csv
6. export_msa_75_OMPK37.csv

For the `antibiotics_OMPK#.tsv` files, they should have the same structure as the `antibiotics.tsv` file in the `test` folder. Each file holds the MIC values for the different isolates that were run through the Amino Acid Annotation pipeline.

The `export_msa_#_OMPK#.csv` files have the same structure as the `export_msa_#.csv` files do in the `test` folder. The number represents the run number ID for the pipeline. The 3 OMPK genes are what references were used against the isolates.

If you would like to run this for yourself, you will need to either generate the files for the 3 OMPK genes, or you will need to go in to the `script/preprocessing` module and change the names of the files to what you wish to input.


## Processed folder
Since the `processed` folder is not included in this repo, and is made programatically, I will say a bit about what you should expect to find in there. The files you should find in this folder are:

1. form_3_ompk35.csv
2. form_3_ompk36.csv
3. form_3_ompk37.csv
4. labels.csv
5. mic_set.csv

For 1-3, these files should be processed to be as described in the paper. Form 3 is an old name given to the type of preprocessing while working in on this study before just using that form and removing forms 1 and 2. 

The `labels.csv` file should be a list of MIC indecies where each row is an isolate and each column is an antibiotic. `0=-1=NA` That means every `0` found in the `labels.csv` file correspond to a `-1` in the `mic_set.csv` file which finally corresponds with a NaN value in the original data. That then means that the given isolate did not have a MIC value for the given antibiotic.

Finally, `mic_set.csv` is a set of MIC values found for all isolate/antibiotic combinations. It will be used to decode the MICs that get returned from the ML algorithms.

## Test folder
The test folder has generated data that is similar to what the real data looks like. It can be used to compare what a user sees to what the paper has. Test output will be given throughout Repo to make sure other people can reproduce the work here.

# File naming scheme and information
For all files, the first column is isolate id so that they can all be joined together later on.

There are 3 types of files:
1. antibiotics.csv
2. export_msa_\<integer>.xlsx

## File type 1
1 above is the list of labels for each antibiotic that will be used in the study. As mentioned in the manuscript, each antibiotic will be a different trained model for each algorithm. The values for each antibiotic/isolate combination is the MIC value. The MIC values are in a range, so if an isolate would have an MIC value above that range then > would be prefixed to the MIC. If the isolate would have a MIC value below the range, it would have <= prefixed.

## File type 2
The integer in this file's name is the run number from the pipeline. The first n rows are the isolates that were run through the pipeline. The last 2 rows correspond with the consensus sequence and reference sequence. The consensus sequence is the consensus at each position for all isolates (including the reference). The reference sequence is taken from the gene fasta reference file given to the pipeline. For each isolate, if a position differs from reference then the amino acid's unique color will be in the cell. If the cell, for the isolate, contains the same amino acid as the reference then there will be no color to the cell. This means nothing for the Machine Learning algorithms (it will be stripped away after being processed), but it was how the pipeline output the file.

There are extra characters that can show up in this file. There are the usual 20 amino acids, but there are more characters. Those are listed below with what they mean:
- (\-) - means the isolate's sequence does not go out this far. It stopped before this position, and this can be thought of as a place holder so that the multiple sequence alignment can be done
- (?) - A frameshift occurred. A frameshift is a particular type of insertion or deletion where there was a number of nucleotides that were inserted or deleted that were not a multiple of 3 (EX: ATTGATGAA -> ATGAA). The new sequence, after the mutation, cannot be converted into codons easily. Because of this, the program inserts a frameshift (fs) mutation at the position the insertion or deletion occurred. The rest of the sequence is removed as it can be thought of the frameshift acting like a stop codon.
- (X) - A stop codon
- (!) - This only occurrs in the consensus sequence, and this means that a consensus could not be reached at that position for given amino acids.

# Preprocessing Deprecation
In the test folder, there is another file `export_<integer>.csv`. This used to be used for creating form 1 during preprocessing. It is left there since the preprocessing Jupyter Notebook still references it. If you would like more detail on why preprocessing forms 1 and 2 are deprecated, please see Notebook `notebooks/data_processing.ipynb` for more information.

