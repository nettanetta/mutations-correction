from transformers import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

MAX_LEN = 512


def get_number_of_different_chars(seq_m, seq_t):
    diff = 0
    min_len = min(len(seq_m), len(seq_t))
    for i in range(min_len):
        if seq_m[i] != seq_t[i]:
            diff += 1
    return diff + abs(len(seq_m) - len(seq_t))


def save_data_stats(orig_fasta, mutated_fasta, output_name):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    stats_dict = defaultdict(list)
    zipped_fasta_lines = zip(SeqIO.parse(mutated_fasta, "fasta"), SeqIO.parse(orig_fasta, "fasta"))
    for record_m, record_t in zipped_fasta_lines:
        m_seq, t_seq = str(record_m.seq), str(record_t.seq)
        tokenized_m = tokenizer(m_seq, padding=True, truncation=True, max_length=MAX_LEN)['input_ids']
        tokenized_t = tokenizer(t_seq, padding=True, truncation=True, max_length=MAX_LEN)['input_ids']

        stats_dict['seq_name'].append(record_m.id)
        stats_dict['len_in_nucleotides'].append(len(m_seq))
        stats_dict['diff_in_nucleotides'].append(get_number_of_different_chars(m_seq, t_seq))
        stats_dict['diff_in_amino_acids'].append(
            get_number_of_different_chars(str(record_m.seq.translate()), str(record_t.seq.translate())))
        stats_dict['diff_in_tokens'].append(get_number_of_different_chars(tokenized_m, tokenized_t))
        stats_dict['len_in_tokens_mutated'].append(len(tokenized_m))
        stats_dict['len_in_tokens_true'].append(len(tokenized_t))
    pd.DataFrame(stats_dict).to_csv(output_name)


if __name__ == "__main__":
    for dataset_type in ['val', 'test', 'train', ]:
        print(f'Processing {dataset_type} dataset')
        save_data_stats(f'/sci/backup/morani/lab/Projects/mutations_detection_temp/data/{dataset_type}_input.fasta',
                        f'/sci/backup/morani/lab/Projects/mutations_detection_temp/data/{dataset_type}_labels.fasta',
                        f'{dataset_type}_stats.csv')

    # with open(train_fasta_mutated) as f:
    #     # for record in SeqIO.parse(f, "fasta"):
    #         # print(tokenizer(str(record.seq),padding=True, truncation=True, max_length=MAX_LEN))
    #         # break
    #     tokens_lengths = [len(tokenizer(str(record.seq),padding=True, truncation=True, max_length=MAX_LEN)['input_ids']) for record in SeqIO.parse(f, "fasta")]
    #     print('max_token_lengths:', max(tokens_lengths))
    #     print('median_token_lengths:', np.median(tokens_lengths))
    #     plt.hist(tokens_lengths, bins=512)
    #     plt.savefig('token_lengths.png')
