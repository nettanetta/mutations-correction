import torch
from transformers import AdamW, get_scheduler, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from dnabert_for_token_classification import BertForTokenClassification
import evaluate
from tqdm import tqdm
from data_handling import MutationDetectionDataset, get_one_hot_encoded_detection_labels
from torch.utils.data import DataLoader

BATCH_SIZE = 32
MAX_LEN = 512
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def model_outputs_from_batch(batch, model, device, tokenizer):
    tokenized_x = tokenizer(batch['mutated_seqs'], padding=True, truncation=True, max_length=MAX_LEN,
                            return_tensors='pt').to(device)
    tokenized_y = tokenizer(batch['orig_seqs'], padding=True, truncation=True, max_length=MAX_LEN,
                            return_tensors='pt').to(device)
    labels = get_one_hot_encoded_detection_labels(tokenized_x, tokenized_y)
    return model(input_ids=tokenized_x['input_ids'], labels=labels), labels


def run_training(train_fasta_m, train_fasta_t, validation_fasta_m, validation_fasta_t, num_epochs=1):
    # Load a pre-trained model and tokenizer
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForTokenClassification(config, num_labels=2).to(DEVICE)

    # in case we have two separate files for train and eval:
    # train_dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer, replacement_flag=True,
    #                                          mutation_rate=0.01)
    # eval_dataset = MutationDetectionDataset(validation_fasta_m, validation_fasta_t, tokenizer, replacement_flag=False)

    # in case we only have one file for both train and eval:
    dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=None)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
    metric = evaluate.load("seqeval")

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            outputs,_ = model_outputs_from_batch(batch, model, device, tokenizer)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            with torch.no_grad():
                outputs,labels = model_outputs_from_batch(batch, model, device, tokenizer)

            predictions = outputs.logits.argmax(dim=-1)
            # labels = batch["labels"]
            # label_list = ['no_mutation', 'mutation']
            # # true_predictions = [
            # #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            # #     for prediction, label in zip(predictions, labels)
            # # ]
            # # true_labels = [
            # #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            # #     for prediction, label in zip(predictions, labels)
            # # ]

            metric.add_batch(predictions=predictions, references=labels)
        # TODO make sure I ignore start, end and padding tokens in the metric
        results = metric.compute()
        print(f"Epoch {epoch + 1}: {results}")

    # Save the model
    model.save_pretrained("path_to_save_model")


if __name__ == "__main__":
    print(f'device is: {DEVICE}')
    train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/gencode.v46.fa'
    train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/gencode.v46.transcripts_fixed.fa'
    run_training(train_fasta_mutated, train_fasta_true, None, None, num_epochs=1)
