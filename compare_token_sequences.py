from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from data_handling import MutationDetectionDataset

if __name__ == "__main__":
    train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/gencode.v46.fa'
    train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/gencode.v46.transcripts_fixed.fa'

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    mutation_detection_dataset = MutationDetectionDataset(train_fasta_mutated, train_fasta_true, tokenizer, verbose=True)

