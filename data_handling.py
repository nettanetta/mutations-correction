from torch.utils.data import Dataset, DataLoader
import torch
from Bio import SeqIO
import random

MAX_LEN = 32  # TODO - change to 512


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


def compare_two_sets_of_sequences(seqs1, seqs2):
    return seqs1['input_ids'] == seqs2['input_ids']


def get_one_hot_encoded_detection_labels(seqs1, seqs2):
    comparison_tensor = (seqs1['input_ids'] != seqs2['input_ids'])
    return torch.nn.functional.one_hot(comparison_tensor.long())



class MutationDetectionDataset(Dataset):

    def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01, verbose=False):
        zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        self.mutated_seqs = []
        self.orig_seqs = []
        for record_m, record_t in zipped_fasta_lines:
            if replacement_flag:
                x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
            else:
                x = record_m.seq
            self.mutated_seqs.append(str(x))
            self.orig_seqs.append(str(record_t.seq))

        # zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        # self.sequences = []
        # self.tokens_labels = []
        # for record_m, record_t in zipped_fasta_lines:
        #     if replacement_flag :
        #         x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
        #     else:
        #         x = record_m.seq
        #     tokenized_x = tokenization_f(str(x), padding=True, truncation=True, max_length=MAX_LEN)['input_ids']
        #     tokenized_y = tokenization_f(str(record_t.seq), padding=True, truncation=True, max_length=MAX_LEN)['input_ids']
        #     self.sequences.append(tokenized_x)
        #     self.tokens_labels.append(compare_two_sequences(tokenized_x, tokenized_y))
        #     if verbose:
        #         print(tokenized_x)
        #         print(tokenized_y)
        #         print(compare_two_sequences(tokenized_x, tokenized_y))
        #         print('----------------')

    def __len__(self):
        return len(self.mutated_seqs)

    def __getitem__(self, idx):
        return {'mutated_seqs': self.mutated_seqs[idx], 'orig_seqs': self.orig_seqs[idx]}
