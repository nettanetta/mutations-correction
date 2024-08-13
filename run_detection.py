import torch
from transformers import AdamW, get_scheduler, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from dnabert_for_token_classification import BertForTokenClassification
import evaluate
import numpy as np
from data_handling import MutationDetectionDataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

BATCH_SIZE = 32
MAX_LEN = 512
SEQ_PAD_TOKEN = 3  # [PAD] token
LABELS_PAD_TOKEN = 0

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# def model_outputs_from_batch(batch, model, device, tokenizer):
#     tokenized_x = tokenizer(batch['mutated_seqs'], padding=True, truncation=True, max_length=MAX_LEN,
#                             return_tensors='pt').to(device)
#     tokenized_y = tokenizer(batch['orig_seqs'], padding=True, truncation=True, max_length=MAX_LEN,
#                             return_tensors='pt').to(device)
#     labels = get_one_hot_encoded_detection_labels(tokenized_x, tokenized_y)
#     return model(input_ids=tokenized_x['input_ids'], labels=labels), labels
#

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


def run_training(train_fasta_m, train_fasta_t, validation_fasta_m, validation_fasta_t, num_epochs=1, learning_rate=2e-5,
                 weight_decay=0.01):
    # Load a pre-trained model and tokenizer
    print('device is: ', DEVICE)

    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertForTokenClassification(config, num_labels=3).to(DEVICE)
    print('total params:', sum(p.numel() for p in model.parameters()))
    print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))




    # in case we only have one file for both train and eval:
    # dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer)
    # train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=None)

    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=detection_collator_func)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True, collate_fn=detection_collator_func)

    # Prepare the data (example using Hugging Face datasets library)
    # Assumes your dataset has been preprocessed appropriately
    # For example: `tokenized_datasets` should be a tokenized dataset object with 'input_ids', 'attention_mask', 'labels'
    lora_config = LoraConfig(target_modules=['query', 'value'], task_type="TOKEN_CLS")
    model = get_peft_model(model, lora_config)
    print('total params after LoRA:', sum(p.numel() for p in model.parameters()))
    print('trainable params loRA:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    args = TrainingArguments(
        output_dir="/sci/backup/morani/lab/Projects/mutations_detection_temp/detection_model",
        logging_dir="/sci/backup/morani/lab/Projects/mutations_detection_temp/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True)

    print(model)
    print('model loaded. loading datasets...')

    # in case we have two separate files for train and eval:
    train_dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer)
    eval_dataset = MutationDetectionDataset(validation_fasta_m, validation_fasta_t, tokenizer)

    print('datasets created. creating trainer...')
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=detection_collator_func,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    print('trainer created. starting training...')
    trainer.train()
    # Define optimizer and learning rate scheduler
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )

    # Move model to GPU if available

    # Define a metric for evaluation
    # metric = evaluate.load("seqeval")

    # Training loop
    # for epoch in range(num_epochs):
    #     # Training
    #     model.train()
    #     for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
    #         outputs = model(input_ids=batch['sequences'].to(device), labels=batch['token_labels'].to(device))
    #         loss = outputs.loss
    #         loss.backward()
    #
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #
    #     # Evaluation
    #     model.eval()
    #     for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
    #         with torch.no_grad():
    #             outputs = model(input_ids=batch['sequences'].to(device), labels=batch['token_labels'].to(device))
    #
    #         predictions = outputs.logits.argmax(dim=-1)
    #         labels = batch['token_labels']
    #
    #         label_list = ['O', 'B', 'I']
    #         str_predictions = [
    #             [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    #             for prediction, label in zip(predictions, labels)
    #         ]
    #         str_labels = [
    #             [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    #             for prediction, label in zip(predictions, labels)
    #         ]
    #
    #         metric.add_batch(predictions=str_predictions, references=str_labels)
    #     # TODO make sure I ignore start, end and padding tokens in the metric
    #     results = metric.compute()
    #     print(f"Epoch {epoch + 1}: {results}")

    # Save the model
    # model.save_pretrained("path_to_save_model")


if __name__ == "__main__":
    print(f'device is: {DEVICE}')
    train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/train_input.fasta'
    train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/train_labels.fasta'
    # train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_input.fasta'
    # train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_labels.fasta'
    validation_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_input.fasta'
    validation_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_labels.fasta'
    # train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/mutated.fa'
    # train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/orig.fa'
    run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3)
