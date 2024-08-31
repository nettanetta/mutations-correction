# mutations-detection
This repo contains the code used for the project "Using DNA Language Models for mutation detection and correction"

Files used for mutation detection:
1. data_handling_for_NER.py - data loader that parses the fasta files etc
2. dnabert_for_token_classification - an implementation of a model for token classification as the authors did not implement such model.
3. run_detection.py - the training loop for mutation detection
4. run_detection_test.py - code for running a pre-trained model on the test set
5. slurm_run_detection.sh - bash script for running the detection on the slurm cluster

Files used for mutation correction:
1. MLM_DNA_BERT.ipynb
2. data_handling_for_MLM.py - data loader that parses the fasta files etc
3. run_correction_train.py  - the training loop for mutation correction

Files used for data analysis and generation:
1. analyze_seqs_stats.ipynb - was used to analyze the changes caused in amino acid sequences and tokenized sequences when simulating mutations.
2. sequences_and_tokens_analysis.py - creates stats on the different datasets
3. split_data.py - splits the data into train, dev, and test.

misc files:
1. mutation_correction_env_1.yml - a yml find containing the requirements for the conda env we used
2. test_stats.csv, train_stats.csv, val_stats.csv - files with stats on the mutated and non-mutated sequences
