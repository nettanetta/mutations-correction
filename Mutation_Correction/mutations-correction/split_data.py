from Bio import SeqIO
import random
import os

def split_fasta(input_fasta, label_fasta, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_dir="data"):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Read sequences from input and label FASTA files
    input_sequences = list(SeqIO.parse(input_fasta, "fasta"))
    label_sequences = list(SeqIO.parse(label_fasta, "fasta"))

    # Ensure both files have the same number of sequences
    assert len(input_sequences) == len(label_sequences), "Input and label FASTA files must have the same number of sequences"

    # Pair input and label sequences
    paired_sequences = list(zip(input_sequences, label_sequences))

    # Shuffle the data
    random.shuffle(paired_sequences)

    # Calculate split sizes
    total_sequences = len(paired_sequences)
    train_size = int(total_sequences * train_ratio)
    val_size = int(total_sequences * val_ratio)

    # Split the data
    train_data = paired_sequences[:train_size]
    val_data = paired_sequences[train_size:train_size + val_size]
    test_data = paired_sequences[train_size + val_size:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to write FASTA files
    def write_fasta(data, input_filename, label_filename):
        input_records, label_records = zip(*data)
        SeqIO.write(input_records, input_filename, "fasta")
        SeqIO.write(label_records, label_filename, "fasta")

    # Write the splits to FASTA files
    write_fasta(train_data, os.path.join(output_dir, "train_input.fasta"), os.path.join(output_dir, "train_labels.fasta"))
    write_fasta(val_data, os.path.join(output_dir, "val_input.fasta"), os.path.join(output_dir, "val_labels.fasta"))
    write_fasta(test_data, os.path.join(output_dir, "test_input.fasta"), os.path.join(output_dir, "test_labels.fasta"))

    print(f"Data split into train, validation, and test sets and saved in '{output_dir}' directory.")

if __name__ == "__main__":
    input_fasta = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/data/gencode.v46.transcripts_fixed.fa'
    label_fasta = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/gencode.v46.fa'
    split_fasta(input_fasta, label_fasta)
