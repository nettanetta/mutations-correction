import torch
from transformers import AdamW, get_scheduler, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from dnabert_for_token_classification import BertForTokenClassification

from data_handling_for_NER import MutationDetectionDataset, detection_collator_func,compute_metrics
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftConfig
import sys


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



def run_training(job_id, train_fasta_m, train_fasta_t, validation_fasta_m, validation_fasta_t, num_epochs=1, learning_rate=2e-5,
                 weight_decay=0.01, use_lora=False,weight_classes=False,ignore_in_labels=False):
    # Load a pre-trained model and tokenizer
    print('device is: ', DEVICE)

    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    if ignore_in_labels:
        num_labels = 2
    else:
        num_labels = 3

    orig_model = BertForTokenClassification(config, num_labels=num_labels, weight_classes=weight_classes).to(DEVICE)
    print('total params:', sum(p.numel() for p in orig_model.parameters()))
    print('trainable params:', sum(p.numel() for p in orig_model.parameters() if p.requires_grad))

    if use_lora:
        lora_config = LoraConfig(target_modules=['query', 'value'], task_type="TOKEN_CLS")
        model = get_peft_model(orig_model, lora_config)
    else:
        model = orig_model
    print('total params after LoRA:', sum(p.numel() for p in model.parameters()))
    print('trainable params loRA:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    args = TrainingArguments(
        output_dir=f"/sci/backup/morani/lab/Projects/mutations_detection_temp/detection_model_{job_id}",
        logging_dir="/sci/backup/morani/lab/Projects/mutations_detection_temp/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True,
        learning_rate=learning_rate)

    print(model)
    print('model loaded. loading datasets...')

    # in case we have two separate files for train and eval:
    train_dataset = MutationDetectionDataset(train_fasta_m, train_fasta_t, tokenizer,ignore_in_labels=ignore_in_labels)
    eval_dataset = MutationDetectionDataset(validation_fasta_m, validation_fasta_t, tokenizer,ignore_in_labels=ignore_in_labels)

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
    # model.save_pretrained(f"/sci/backup/morani/lab/Projects/mutations_detection_temp/manual_saves/{job_id}/lora_model")


if __name__ == "__main__":
    print(f'device is: {DEVICE}')
    # parse arguments
    if len(sys.argv) == 2:
        _, job_id = sys.argv
    else:
        job_id = 'test'



    # # for real runs
    # train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/train_input.fasta'
    # train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/train_labels.fasta'
    # validation_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_input.fasta'
    # validation_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/val_labels.fasta'

    # for real runs on simple data
    train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/simple_train_input.fasta'
    train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/simple_train_labels.fasta'
    validation_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/simple_val_input.fasta'
    validation_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/simple_val_labels.fasta'

    # # for test runs
    # train_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/mutated_1000_seqs.fa'
    # train_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/orig_1000_seqs.fa'
    # validation_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/mutated_1000_seqs.fa'
    # validation_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/orig_1000_seqs.fa'

    # use_lora=False,weight_classes=False,ignore_in_labels=False
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3)
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3,use_lora=True)
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3,use_lora=True, weight_classes=True)
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3, use_lora=True, weight_classes=True, ignore_in_labels=True)
    # no lora, weighted classes, without "2" labels
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3, use_lora=False, weight_classes=True, ignore_in_labels=True)
    # no lora, weighted classes, with "2" labels
    # run_training(train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3,use_lora=False, weight_classes=True, ignore_in_labels=False)
    # lora, weighted classes, without "2" labels, faster learning rate
    # run_training(job_id, train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true, num_epochs=3,
    #              use_lora=False, weight_classes='effective_num_of_samples', ignore_in_labels=True, learning_rate=5e-4) #22936323
    # run_training(job_id, train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true,
    #              num_epochs=3,
    #              use_lora=False, weight_classes='inverse', ignore_in_labels=True, learning_rate=5e-4) # 22936322
    # run_training(job_id, train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true,
    #              num_epochs=3,
    #              use_lora=False, weight_classes='effective_num_of_samples', ignore_in_labels=True,
    #              learning_rate=5e-5)  # 22936684
    run_training(job_id, train_fasta_mutated, train_fasta_true, validation_fasta_mutated, validation_fasta_true,
                 num_epochs=10,
                 use_lora=False, weight_classes='inverse', ignore_in_labels=True,
                 learning_rate=5e-5)  # 22936725
