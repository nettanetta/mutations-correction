from torch.utils.data import Dataset, DataLoader
import torch
from Bio import SeqIO
import random
import data_handling_for_MLM as dhmlm
import evaluate
import numpy as np

MAX_LEN = 512
SEQ_PAD_TOKEN = 3  # [PAD] token
SEQ_END_TOKEN = 2  #
LABELS_PAD_TOKEN = 0


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


def detection_collator_func(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    max_len = max([len(t) for t in input_ids])

    input_ids = [torch.cat((t, torch.ones(max_len - len(t), dtype=torch.long) * SEQ_PAD_TOKEN)) for t in input_ids]
    labels = [torch.cat((t, torch.ones(max_len - len(t), dtype=torch.long) * LABELS_PAD_TOKEN)) for t in labels]

    return {'input_ids': torch.stack(input_ids), 'labels': torch.stack(labels)}


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    metric = evaluate.load("seqeval")

    predictions = np.argmax(logits, axis=2)
    label_list = ['O', 'B', 'I']
    str_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    str_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return metric.compute(predictions=str_predictions, references=str_labels)


#
# # def compare_two_sets_of_sequences(seqs1, seqs2):
# #     return seqs1['input_ids'] == seqs2['input_ids']
#
#
# def get_one_hot_encoded_detection_labels(seqs1, seqs2):
#     comparison_tensor = (seqs1!= seqs2)
#     return torch.nn.functional.one_hot(comparison_tensor.long())

def get_iob1_labels(seqs1, seqs2, in_labels_val=2):
    comparison_tensor = (seqs1 != seqs2)
    labels = torch.zeros_like(seqs1)

    begin_mask = (comparison_tensor == 1) & (torch.cat((torch.tensor([0]), comparison_tensor[:-1])) == 0)
    labels[begin_mask] = 1

    in_mask = (comparison_tensor == 1) & (labels == 0)
    labels[in_mask] = in_labels_val
    # labels[in_mask] = -100 #2 # can change this to -100 if we want to ignore in labels (count each mutation once)

    special_tokens_mask = ((seqs1 == SEQ_PAD_TOKEN) & (seqs2 == SEQ_PAD_TOKEN)) | (
            (seqs1 == SEQ_END_TOKEN) & (seqs2 == SEQ_END_TOKEN))
    labels[special_tokens_mask] = -100
    labels[0] = -100

    return labels


class MutationDetectionDataset(Dataset):

    def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01, verbose=False,
                 ignore_in_labels=False):
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
            if replacement_flag:
                x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
            else:
                x = record_m.seq
            tokenized_x = \
            tokenization_f(str(x), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')[
                'input_ids'].squeeze(0)
            tokenized_y = \
            tokenization_f(str(record_t.seq), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')[
                'input_ids'].squeeze(0)
            tokenized_x, tokenized_y = dhmlm.pad_sequences(tokenized_x, tokenized_y)
            self.sequences.append(tokenized_x)
            if ignore_in_labels:
                in_labels_val = -100
            else:
                in_labels_val = 2
            self.token_labels.append(get_iob1_labels(tokenized_x, tokenized_y, in_labels_val))
            if verbose:
                print(self.sequences[-1])
                print(self.token_labels[-1])
                print('----------------')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'input_ids': self.sequences[idx], 'labels': self.token_labels[idx]}
