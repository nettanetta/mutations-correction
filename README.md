# mutations-detection
the file `run_detection.py` should run the training an evaluating for the mutation correction pipeline.
dnabert_for_sequence_classification.py has the code adapting DNABERT for the task of sequence classification because it was not implemented by DNABERT's authors.
data_handling.py has the code for creating a pytorch dataset based on fasta files - you can use it either using a single fasta file and adding mutations to it by random (that's what I currently did) or giving it two fasta files, one with and without mutations. 

the code is not running properly because of environment issues. but this should be the general instuctions for running the mutation detection: 
1. You need to create a conda env based on this mutation_correction_env.yml 
2. to run the code on the cluster, start with creating an interactive run with GPUs:
    srun --gres=gpu:a5000 --mem=8G --pty $SHELL
3. conda activate mutation_correction_env_1
4. module load cuda/11.8 
5. module load nvidia
6. python3 run_detection.py

dnabert2_python38 worked for running compare_token_sequences.py