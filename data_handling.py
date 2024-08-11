from torch.utils.data import Dataset, DataLoader
import torch
from Bio import SeqIO
import random
MAX_LEN = 32 # TODO - change to 512

def insert_single_replacement(seq, mutation_rate):
    new_seq = []
    mutation_added = False
    for c in seq:
        if mutation_added == False and random.random() < mutation_rate:
            choice_str = list({'A', 'C', 'G', 'T'} - {c})
            new_seq.append(random.choice(choice_str))
        else:
            new_seq.append(c)
    return ''.join(new_seq)


def get_onehot_for_first_missmatch(seq1, seq2):
    for index, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            return [1 if i == index else 0 for i in range(len(seq1))]
    return [0] * len(seq1)

def compare_two_sequences(seq1, seq2):
    return [1 if a != b else 0 for index, (a, b) in enumerate(zip(seq1, seq2))]

class MutationDetectionDataset(Dataset):

    def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01, verbose=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        self.sequences = []
        self.tokens_labels = []
        for record_m, record_t in zipped_fasta_lines:
            if replacement_flag :
                x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
            else:
                x = record_m.seq
            tokenized_x = tokenization_f(str(x), padding=True, truncation=True, max_length=MAX_LEN)['input_ids']
            tokenized_y = tokenization_f(str(record_t.seq), padding=True, truncation=True, max_length=MAX_LEN)['input_ids']
            self.sequences.append(tokenized_x)
            self.tokens_labels.append(compare_two_sequences(tokenized_x, tokenized_y))
            if verbose:
                print(tokenized_x)
                print(tokenized_y)
                print(compare_two_sequences(tokenized_x, tokenized_y))
                print('----------------')


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.sequences[idx]), 'labels': torch.tensor(self.tokens_labels[idx])}
