# Input Data
There should be 2 folders in this section: `main` and `processed`.

## Main
There are 2 main things that should be in the `main` folder.

### OUT
There should be an `OUT` folder within main (so a folder path of `main/OUT/`). This will hold fasta files that need to be 
processed. These files will need to be in the form `Sentry-{year}-{isolate_index}_contigs.fasta`
* `year` - The year the data was collected
* `isolate_index` - The index for the isolate

### CSV file
There should be a `mic_data.csv` file also in `main` folder. This should have columns:
1. `index` - the indicies for each isolate
2. `Study Year` - The year the isolate was collected
3. The rest of the columns are all antibiotics where their cells are MICs for the given isolates

## Processed
This folder must be created as the different script files will be placing processed files here. The list of files that will be
added to this folder are:
1. train.libsvm (for the XGBoost algorithm training)
2. tesst.libsvm (used for all model testing)
3. nn_train (80% from train.libsvm formatted for nn)
4. nn_val (20% from train.libsvm formatted for nn)
5. nn_test (parsed from test.libsvm to test nn)
6. kmers_and_antibiotics.csv (ids for all kmers and antibiotics used in libsvm files)
7. control_test.libsvm (test.libsvm parsed for antibiotics that the control model knows)