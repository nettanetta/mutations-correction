from torch.utils.data import Dataset, DataLoader
import torch
from Bio import SeqIO
import random
import data_handling_for_MLM as dhmlm

MAX_LEN = 512
SEQ_PAD_TOKEN = 3 # [PAD] token
SEQ_END_TOKEN = 2 #
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

#
# # def compare_two_sets_of_sequences(seqs1, seqs2):
# #     return seqs1['input_ids'] == seqs2['input_ids']
#
#
# def get_one_hot_encoded_detection_labels(seqs1, seqs2):
#     comparison_tensor = (seqs1!= seqs2)
#     return torch.nn.functional.one_hot(comparison_tensor.long())

def get_iob1_labels(seqs1, seqs2):
    comparison_tensor = (seqs1!= seqs2)
    labels = torch.zeros_like(seqs1)

    begin_mask = (comparison_tensor == 1) & (torch.cat((torch.tensor([0]), comparison_tensor[:-1])) == 0)
    in_mask = (comparison_tensor == 1) & (labels == 0)
    special_tokens_mask = ((seqs1 == SEQ_PAD_TOKEN) & (seqs2 == SEQ_PAD_TOKEN)) | ((seqs1 == SEQ_END_TOKEN) & (seqs2 == SEQ_END_TOKEN))

    labels[begin_mask] = 1
    labels[in_mask] = 2 # can change this to -100 if we want to ignore in labels (count each mutation once)
    labels[special_tokens_mask] = -100
    labels[0] = -100

    return labels


class MutationDetectionDataset(Dataset):

    def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01, verbose=False):
        # zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        # self.mutated_seqs = []
        # self.orig_seqs = []
        # for record_m, record_t in zipped_fasta_lines:
        #     if replacement_flag:
        #         x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
        #     else:
        #         x = record_m.seq
        #     self.mutated_seqs.append(str(x))
        #     self.orig_seqs.append(str(record_t.seq))

        zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        self.sequences = []
        self.token_labels = []
        for record_m, record_t in zipped_fasta_lines:
            if replacement_flag :
                x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
            else:
                x = record_m.seq
            tokenized_x = tokenization_f(str(x), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
            tokenized_y = tokenization_f(str(record_t.seq), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
            tokenized_x, tokenized_y = dhmlm.pad_sequences(tokenized_x, tokenized_y)
            self.sequences.append(tokenized_x)
            self.token_labels.append(get_iob1_labels(tokenized_x, tokenized_y))
            if verbose:
                print(self.sequences[-1])
                print(self.token_labels[-1])
                print('----------------')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'sequences': self.sequences[idx], 'token_labels': self.token_labels[idx]}
