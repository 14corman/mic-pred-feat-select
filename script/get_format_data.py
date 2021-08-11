# -*- coding: utf-8 -*-
"""
Created on Jun 30 8:14 PM 2021

@author: Cory Kromer-Edwards

Gets fasta data from data/main/OUT/ and combine that with mics from data/main/mic_data.csv.
1. Find all isolates within csv that also have fasta file in OUT
2. Take those fasta files and run them through KMC to count k-mers of size 10
3. Put sequences into a set data type to get unique k-mers
4. Get list of sorted, unique kmers and write that, and antibiotics, to csv file for reference later
5. Put K-Mer data randomly into either train or test libsvm file

REQUIRES that fasta files be in the form: Sentry-{year}-{col_num}_contigs.fasta
year: the year the data was collected
col_num: an index found in the mic_data.csv

REQUIRES that mic_data.csv has:
1. a column with header being "Study Year" and each element in that column being in [2016, 2017, 2018, 2019, 2020]
2. A numerical, unique index column
3. All antibiotics being columns that make up the following list: ["Aztreonam", "Ceftazidime-avibactam",
    "Piperacillin-tazobactam", "Ampicillin-sulbactam", "Ceftolozane-tazobactam", "Cefepime", "Ceftaroline",
    "Ceftazidime", "Ceftobiprole", "Ceftriaxone", "Imipenem", "Doripenem", "Meropenem"]
4. Isolates are rows, so the index corresponds to some isolate id
"""
import subprocess
import shlex
import os
import pandas as pd
from pathlib import Path
import math
import random
import sys

DATA_DIR = f"../data/main"
PROCESSED_DIR = f"../data/processed"
FASTA_DIR = f"{DATA_DIR}/OUT"

ANTIBIOTICS = ["Aztreonam", "Ceftazidime-avibactam", "Piperacillin-tazobactam", "Ampicillin-sulbactam",
               "Ceftolozane-tazobactam",
               "Cefepime", "Ceftaroline", "Ceftazidime", "Ceftobiprole", "Ceftriaxone", "Imipenem", "Doripenem",
               "Meropenem"]


def _run_cmd(cmd):
    """
    Original code taken from: https://janakiev.com/blog/python-shell-commands/
    :param cmd: String linux command to use
    :return: 0 if command succeeded, -1 otherwise
    """
    with subprocess.Popen(shlex.split(cmd),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True) as process:

        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code == 0:
                    return 0
                else:
                    # Process has finished, read rest of the output
                    for error in process.stderr.readlines():
                        print(error.strip())
                    return -1


def _run_kmc(contig_fasta_file):
    """
    This runs KMC given input seen from:
    https://github.com/PATRIC3/mic_prediction/blob/master/mic_prediction_fasta.sh

    The output (a file named {contig_fasta_file}.10.kmrs) will be parsed and returned.
    :param contig_fasta_file: The fasta file to get k-mer counts from
    :return: A dictionary of k-mers and their counts
    """
    cmd = f"kmc -k10 -fm -ci1 -cs1677215 {FASTA_DIR}{contig_fasta_file} {FASTA_DIR}{os.path.splitext(contig_fasta_file)[0]} {FASTA_DIR} "
    output_file = f"{FASTA_DIR}{os.path.splitext(contig_fasta_file)[0]}.10.kmrs"

    fasta_path = Path(output_file)
    if fasta_path.is_file():  # If the file has already been run through kmc, use precompiled file as cache
        with open(output_file) as kmer_file:
            kmers = dict()
            for line in kmer_file:
                parts = line.strip().split('\t')
                # print(parts, end="\r")    # For debugging purposes
                if len(parts) > 1:
                    kmers[parts[0]] = parts[1]

        return kmers
    else:  # File has not been run through kmc yet, run through and get results
        result = _run_cmd(cmd)
        if result == 0:
            dump_cmd = f"kmc_dump {FASTA_DIR}{os.path.splitext(contig_fasta_file)[0]} {output_file}"
            result = _run_cmd(dump_cmd)
            if result == 0:
                with open(output_file) as kmer_file:
                    kmers = dict()
                    for line in kmer_file:
                        parts = line.strip().split('\t')
                        if len(parts) > 1:
                            kmers[parts[0]] = parts[1]

                return kmers
            else:
                raise Exception(f"KMC dump failed")
        else:
            raise Exception(f"KMC command failed")


def collect_data(input_type):
    mic_df = pd.read_csv(f'{DATA_DIR}/mic_data.csv', index_col=0)
    unique_kmers = set()
    # This file should already exist. If you are restarting with fresh data, make sure the file generated
    # matches this file. If not, you will also have to retrain XGBoost to get new selected features.
    if not Path(f"{PROCESSED_DIR}/kmers_and_antibiotics.csv").is_file():
        for year in [2016, 2017, 2018, 2019, 2020]:
            print(f"Starting year retreival: {year}")
            mic_year_df = mic_df.loc[mic_df['Study Year'] == year]

            # For print out
            num_rows = len(mic_year_df.index)
            num_numbers = len(str(num_rows))
            # 1. Find all isolates within csv that also have fasta file in OUT
            for index, col_num in enumerate(mic_year_df.index):
                print(f"{str(index).rjust(num_numbers, '0')}/{num_rows}", end='\r')
                fasta_name = f"Sentry-{year}-{col_num}_contigs.fasta"
                fasta_path = Path(f"{DATA_DIR}/{fasta_name}")
                if fasta_path.is_file():
                    # 2. Take those fasta files and run them through KMC to count k-mers of size 10
                    output = _run_kmc(fasta_name)
                    # 3. Put sequences into a set data type to get unique k-mers
                    unique_kmers.update(list(output.keys()))

            # 4. Get list of sorted, unique kmers and write that, and antibiotics, to csv file for reference later
            sorted_uniq_kmers = list(unique_kmers)
            sorted_uniq_kmers.sort()
            with open(f"{PROCESSED_DIR}/kmers_and_antibiotics.csv", 'w') as file:
                file.write("id,feature\n")
                for i, kmer in enumerate(sorted_uniq_kmers):
                    file.write(f"{i},{kmer}\n")

                n_kmers = len(sorted_uniq_kmers)
                for i, ant in enumerate(ANTIBIOTICS):
                    file.write(f"{i + n_kmers},{ant}\n")

            feature_dict = pd.read_csv(f"{PROCESSED_DIR}/kmers_and_antibiotics.csv", index_col=1).to_dict().get('id')
    else:
        feature_dict = pd.read_csv(f"{PROCESSED_DIR}/kmers_and_antibiotics.csv", index_col=1).to_dict().get('id')

    # 5. Put K-Mer data randomly into either train or test libsvm file
    with open(f"{PROCESSED_DIR}/train.libsvm", 'w') as train_file, open(f"{PROCESSED_DIR}/test.libsvm",
                                                                        'w') as test_file:
        for year in [2016, 2017, 2018, 2019, 2020]:
            print(f"Starting year collection: {year}")
            mic_year_df = mic_df.loc[mic_df['Study Year'] == year]
            mics = mic_year_df[ANTIBIOTICS]

            # For print out
            num_rows = len(mic_year_df.index)
            num_numbers = len(str(num_rows))
            for index, col_num in enumerate(mic_year_df.index):
                print(f"{str(index).rjust(num_numbers, '0')}/{num_rows}", end='\r')
                fasta_name = f"Sentry-{year}-{col_num}_contigs.fasta"
                fasta_path = Path(f"{FASTA_DIR}/{fasta_name}")
                if fasta_path.is_file():
                    kmer_string = ''
                    output = _run_kmc(fasta_name)
                    for kmer in sorted(feature_dict.keys()):
                        if kmer in output:
                            kmer_string += f"{feature_dict[kmer]}:{output[kmer]} "

                    for mic, (ant_id, ant) in zip(mics.loc[col_num].values, enumerate(ANTIBIOTICS)):
                        mic = str(mic).replace('<=', '').replace('>', '')
                        if mic != "" and not math.isnan(float(mic)):
                            # +13 to make sure all log_2(mic) >= 0
                            tmp_kmer_string = f"{round(math.log2(float(mic))) + 13} {kmer_string}{feature_dict[ant]}:1\n"

                            if input_type == 0:
                                # Make Train test files shuffled have 80/20 split.
                                # Data is shuffled as it can randomly be placed into train or test during processing
                                if random.random() < 0.8:
                                    train_file.write(tmp_kmer_string)
                                else:
                                    test_file.write(tmp_kmer_string)
                            else:
                                test_file.write(tmp_kmer_string)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        collect_data(0)

    run_command = sys.argv[1]
    if run_command == "redo":
        collect_data(0)
    elif run_command == "test":
        collect_data(1)
    else:
        raise ValueError(f"Unknown input '{run_command}'. Expected either no input or 1 of ['redo', 'test']")
    collect_data()
