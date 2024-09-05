import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from data_handling_for_MLM import MutationDetectionDataset, collate_fn
from torch.utils.data import DataLoader
import Levenshtein
import numpy as np

BATCH_SIZE = 4096
MAX_LEN = 512
SEQ_PAD_TOKEN = 3 # [PAD] token
LABELS_PAD_TOKEN = 0
NUM_EPOCHS = 3
LR = 5e-5
WEIGTH_DECAY = 0.01


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def compute_median_edit_distance(edit_distances):
    return np.median(edit_distances)


def compute_mean_edit_distance(edit_distances):
    return np.mean(edit_distances)


def compute_normalized_mean_edit_distance(edit_distances, label_texts):
    # Calculate normalized mean edit distance
    normalized_edit_distances = [edit_distance / len(label_text) for edit_distance, label_text in zip(edit_distances, label_texts)]
    return np.mean(normalized_edit_distances)


def compute_metrics(eval_pred):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Convert predictions and labels to lists of token IDs
    predictions = predictions.tolist()
    labels = labels.tolist()
    
    # Decode predictions and labels, filtering out invalid labels
    pred_texts = []
    label_texts = []
    for pred, label in zip(predictions, labels):
        # Filter out -100 (or any invalid token IDs) from labels
        valid_label_ids = [id for id in label if id != -100]
        # Decode only valid token IDs
        pred_texts.append(tokenizer.decode(pred, skip_special_tokens=True))
        label_texts.append(tokenizer.decode(valid_label_ids, skip_special_tokens=True))
    
    # Calculate Levenshtein distance
    edit_distances = [Levenshtein.distance(pred, label) for pred, label in zip(pred_texts, label_texts)]
    
    
    return {"avg_edit_distance": compute_mean_edit_distance(edit_distances),
            "median_edit_distance": compute_median_edit_distance(edit_distances),
            "normalized_avg_edit_distance": compute_normalized_mean_edit_distance(edit_distances, labels)}



def run_correction(train_fasta_mutated, train_fasta_true, validation_fasta_m, validation_fasta_t, num_epochs=1):
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForMaskedLM(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGTH_DECAY)


    train_mutation_dataset = MutationDetectionDataset(train_fasta_mutated, train_fasta_true, tokenizer, verbose=False)
    validation_mutation_dataset = MutationDetectionDataset(validation_fasta_m, validation_fasta_t, tokenizer, verbose=False)
    print('Dataset created')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/correction/results",  # output directory
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=4_000,
        save_total_limit=3,
        logging_dir='./models/correction/logs',
        evaluation_strategy="epoch",  # Ensure evaluations occur
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_mutation_dataset,
        eval_dataset=validation_mutation_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    print(f'device is: {DEVICE}')
    # train_fasta_mutated = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/data/gencode.v46.transcripts_fixed.fa'
    # train_fasta_true = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/gencode.v46.fa'
    train_fasta_mutated = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/mutations-correction/data/train_input.fasta'
    train_fasta_true = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/mutations-correction/data/train_labels.fasta'
    validation_fasta_m = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/mutations-correction/data/val_input.fasta'
    validation_fasta_t = '/ems/elsc-labs/habib-n/yuval.rom/school/ANLP/final_project/mutations-correction/data/val_labels.fasta'
    run_correction(train_fasta_mutated, train_fasta_true, validation_fasta_m=train_fasta_mutated, validation_fasta_t=train_fasta_true, num_epochs=NUM_EPOCHS)
