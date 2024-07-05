import torch
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from transformers.models.bert.configuration_bert import BertConfig
from dnabert_for_token_classification import BertForTokenClassification
from Bio import SeqIO
import random
from datasets import load_metric
from tqdm import tqdm
from data_handling import MutationDetectionDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1024

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def run_training(train_fasta_m, train_fasta_t, validation_fasta_m, validation_fasta_t,num_epochs=3):
    # Load a pre-trained model and tokenizer
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForTokenClassification(config, num_labels=1).to(DEVICE)

    train_dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer, replacement_flag=True,
                                             mutation_rate=0.01)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    eval_dataset = MutationDetectionDataset(validation_fasta_m, validation_fasta_t, tokenizer, replacement_flag=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

    # Prepare the data (example using Hugging Face datasets library)
    # Assumes your dataset has been preprocessed appropriately
    # For example: `tokenized_datasets` should be a tokenized dataset object with 'input_ids', 'attention_mask', 'labels'

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device is: ', device)
    model.to(device)

    # Define a metric for evaluation
    metric = load_metric("seqeval")

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch.keys())
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Remove ignored index (special tokens) for evaluation
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(f"Epoch {epoch + 1}: {results}")

    # Save the model
    model.save_pretrained("path_to_save_model")



if __name__ == "__main__":
    print(f'device is: {DEVICE}')
    run_training()
