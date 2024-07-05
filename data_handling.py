from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import random
def insert_single_replacement(seq, mutation_rate):
    new_seq = []
    mutation_added=False
    for c in seq:
        if mutation_added==False and random.random() < mutation_rate:
            choice_str = list({'A', 'C', 'G', 'T'} - {c})
            new_seq.append(random.choice(choice_str))
        else:
            new_seq.append(c)
    return ''.join(new_seq)

def get_onehot_for_first_missmatch(seq1, seq2):
    for index, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            return [1 if i == index else 0 for i in range(len(seq1))]
    return None


class MutationDetectionDataset(Dataset):

    def __init__(self, fasta_m, fasta_t, tokenization_f, replacement_flag=False, mutation_rate=0.01):
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
            tokenized_x = tokenization_f(x)
            tokenized_y = tokenization_f(record_t.seq)
            self.sequences.append(tokenized_x)
            self.tokens_labels.append(get_onehot_for_first_missmatch(tokenized_x, tokenized_y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {'input_ids': self.sequences[idx], 'labels': self.tokens_labels[idx]}
