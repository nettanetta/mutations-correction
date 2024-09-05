# from torch.utils.data import Dataset, DataLoader
# import torch
# from Bio import SeqIO
# import random


# MAX_LEN = 512
# MASK_TOKEN = 4 # [MASK] token
# PAD_TOKEN = 3 # [PAD] token
# ENDING_TOKEN = 2 # [SEP] token


# def insert_single_replacement(seq, mutation_rate):
#     new_seq = []
#     mutation_added = False
#     for c in seq:
#         if mutation_added == False and random.random() < mutation_rate:
#             choice_str = list({'A', 'C', 'G', 'T'} - {c})
#             new_seq.append(random.choice(choice_str))
#         else:
#             new_seq.append(c)
#     return ''.join(new_seq)


# def get_onehot_for_first_missmatch(seq1, seq2):
#     for index, (a, b) in enumerate(zip(seq1, seq2)):
#         if a != b:
#             return [1 if i == index else 0 for i in range(len(seq1))]
#     return [0] * len(seq1)


# def pad_sequences(seq1, seq2, max_len=MAX_LEN, pad_token=PAD_TOKEN):
#     max_len = max(len(seq1), len(seq2))
#     if seq1.shape[0] < max_len:
#         pad_vector = torch.ones(max_len - len(seq1), dtype=torch.long) * pad_token
#         seq1 = torch.cat((seq1, pad_vector))

#     elif seq2.shape[0] < max_len:
#         pad_vector = torch.ones(max_len - len(seq2), dtype=torch.long) * pad_token
#         seq2 = torch.cat((seq2, pad_vector))   

#     assert seq1.shape == seq2.shape
#     return seq1, seq2


# def compare_two_sequences(seq1, seq2):
#     return seq1 != seq2


# def mask_sequence(seq, mask_vector):
#     seq[mask_vector] = MASK_TOKEN # [MASK] token
#     return seq

# def mask_labels(seq, mask_vector):
#     seq[~mask_vector] = -100 # ignore token
#     return seq


# def collate_fn(batch):
#     input_ids = [item['input_ids'] for item in batch]
#     labels = [item['labels'] for item in batch]
#     max_len = max([len(t) for t in input_ids])

#     input_ids = [torch.cat((t, torch.ones(max_len - len(t), dtype=torch.long) * PAD_TOKEN)) for t in input_ids]
#     labels = [torch.cat((t, torch.ones(max_len - len(t), dtype=torch.long) * PAD_TOKEN)) for t in labels]

#     return {'input_ids': torch.stack(input_ids), 'labels': torch.stack(labels)}


# def fix_tokens_after_k_changes(original_tokens, mutated_tokens, tokenizer, k):
#     # Ensure the tensors have the same length
#     assert original_tokens.size() == mutated_tokens.size(), "Original and mutated sequences must be the same length"

#     # Initialize counters and result tensor
#     changes_count = 0
#     fixed_tokens = mutated_tokens.clone()

#     # Iterate over the tokens
#     for i in range(len(original_tokens)):
#         if original_tokens[i] != mutated_tokens[i]:
#             changes_count += 1
#         else: # less than k consecutive tokens have been changed
#             changes_count = 0

#         # If consecutive changes exceed k, use original token
#         if changes_count > k:
#             fixed_tokens[i] = original_tokens[i]

#     return fixed_tokens


# class MutationDetectionDataset(Dataset):

#     def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01, verbose=False, k=1):
 
#         zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
#         self.sequences = []
#         self.tokens_labels = []
#         self.tokens_gt = []
#         self.mutations = []
#         for record_m, record_t in zipped_fasta_lines:
#             if replacement_flag :
#                 x = insert_single_replacement(record_m.seq, mutation_rate=mutation_rate)
#             else:
#                 x = record_m.seq
#             tokenized_x = tokenization_f(str(x), padding=False, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
#             tokenized_y = tokenization_f(str(record_t.seq), padding=False, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
#             tokenized_x, tokenized_y = pad_sequences(tokenized_x, tokenized_y)
#             fixed_tokens = fix_tokens_after_k_changes(tokenized_y, tokenized_x, tokenization_f, k=k)
#             mask_vector = compare_two_sequences(fixed_tokens, tokenized_y)
#             tokenized_x = mask_sequence(fixed_tokens, mask_vector)
#             self.sequences.append(tokenized_x)
#             self.tokens_gt.append(tokenized_y)
#             tokenized_labels = mask_labels(tokenized_y.clone(), mask_vector)
#             self.tokens_labels.append(tokenized_labels)
#             if verbose:
#                 print('x', tokenized_x)
#                 print('labels_mask', tokenized_labels)
#                 print('y', tokenized_y)
#                 print(tokenized_x.shape == tokenized_y.shape)                
#                 # print('compare', mask_vector)
#                 print('----------------')

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         assert self.sequences[idx].shape == self.tokens_labels[idx].shape
#         return {'input_ids': self.sequences[idx], 'labels': self.tokens_labels[idx], 'gt': self.tokens_gt[idx]}


import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import random

MAX_LEN = 512
MASK_TOKEN = 4  # [MASK] token
PAD_TOKEN = 3  # [PAD] token
ENDING_TOKEN = 2  # [SEP] token

def insert_single_replacement(seq, mutation_rate):
    new_seq = list(seq)
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            new_seq[i] = random.choice(list({'A', 'C', 'G', 'T'} - {new_seq[i]}))
    return ''.join(new_seq)

def pad_sequence(seq, max_len=MAX_LEN, pad_token=PAD_TOKEN):
    if len(seq) < max_len:
        return torch.cat((seq, torch.ones(max_len - len(seq), dtype=torch.long) * pad_token))
    return seq[:max_len]

def compare_two_sequences(seq1, seq2):
    return seq1 != seq2

def mask_sequence(seq, mask_vector, mask_prob=1.0):
    mask = mask_vector & (torch.rand(mask_vector.shape) < mask_prob)
    seq[mask] = MASK_TOKEN
    return seq

class MutationDetectionDataset(Dataset):
    def __init__(self, fasta_m, fasta_t, tokenizer, replacement_flag=False, mutation_rate=0.01, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        
        zipped_fasta_lines = zip(SeqIO.parse(fasta_m, "fasta"), SeqIO.parse(fasta_t, "fasta"))
        for record_m, record_t in zipped_fasta_lines:
            if replacement_flag:
                x = insert_single_replacement(str(record_m.seq), mutation_rate=mutation_rate)
            else:
                x = str(record_m.seq)
            
            tokenized_x = self.tokenizer(x, padding=False, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
            tokenized_y = self.tokenizer(str(record_t.seq), padding=False, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'].squeeze(0)
            
            tokenized_x = pad_sequence(tokenized_x)
            tokenized_y = pad_sequence(tokenized_y)
            
            mask_vector = compare_two_sequences(tokenized_x, tokenized_y)
            masked_input = mask_sequence(tokenized_x.clone(), mask_vector, mask_prob)
            
            labels = tokenized_y.clone()
            labels[~mask_vector] = -100
            
            attention_mask = (tokenized_x != PAD_TOKEN).long()
            
            self.input_ids.append(masked_input)
            self.labels.append(labels)
            self.attention_mask.append(attention_mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'attention_mask': self.attention_mask[idx]
        }
