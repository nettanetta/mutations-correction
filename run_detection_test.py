from dnabert_for_token_classification import BertForTokenClassification
from data_handling_for_NER import MutationDetectionDataset, detection_collator_func, compute_metrics
from transformers import AutoModel, Trainer, TrainingArguments, BertConfig, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy.special import softmax
from peft import PeftModel, PeftConfig

BATCH_SIZE = 64
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def run_detection_on_test_dataset(model_paths, test_fasta_m, test_fasta_t, ignore_in_labels):
    # config = PeftConfig.from_pretrained(model_path)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    # print('loading model...')
    for model_path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dataset = MutationDetectionDataset(test_fasta_m, test_fasta_t, tokenizer, ignore_in_labels=ignore_in_labels)
        print(f'loading model from {model_path}')
        model = BertForTokenClassification.from_pretrained(model_path, config=config, local_files_only=True).to(DEVICE)
        # print('adding LoRA...')
        # model = PeftModel.from_pretrained(model, model_path)
        model.eval()


        args = TrainingArguments(
            output_dir=model_path,
            # logging_dir="/sci/backup/morani/lab/Projects/mutations_detection_temp/logs",
            # evaluation_strategy="epoch",
            # save_strategy="epoch",
            # num_train_epochs=num_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            fp16=True)

        trainer = Trainer(
            model=model,
            args=args,
            # train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            data_collator=detection_collator_func,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        predictions = trainer.predict(dataset)
        # np.savetxt(f'{model_path}/predictions.txt', np.argmax(predictions.predictions, axis=2))

        # draw a precision-recall curve
        labels_no_skips = []
        probabilities_no_skips = []
        for preds, single_seq_labels in zip(predictions.predictions, dataset.token_labels):
            for i, label in enumerate(single_seq_labels):
                if label != -100:
                    labels_no_skips.append(int(label))
                    probabilities_no_skips.append(softmax(preds[i]))

        precision, recall, thresholds = precision_recall_curve(labels_no_skips,
                                                               softmax(np.array(probabilities_no_skips), axis=1)[:, 1])
        plt.plot(recall, precision)
        plt.scatter(recall, precision, color='orange')
        xlim_extra = 0.1
        plt.xlim([0.0 - xlim_extra, 1.0 + xlim_extra])
        plt.ylim([0.0 - xlim_extra, 1.0 + xlim_extra])
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Train Precision-Recall curve")
        plt.savefig(f'{model_path}/precision_recall_curve.png')
        plt.clf()



if __name__ == '__main__':
    # lora_model = AutoModel.from_pretrained("/sci/backup/morani/lab/Projects/mutations_detection_temp/manual_saves/test/lora_model")

    # # just for testing
    # test_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/mutated_1000_seqs.fa'
    # test_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/orig_1000_seqs.fa'
    # real data
    test_fasta_mutated = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/test_input.fasta'
    test_fasta_true = '/sci/backup/morani/lab/Projects/mutations_detection_temp/data/test_labels.fasta'
    ignore_in_labels = True

    model_paths = ['/sci/backup/morani/lab/Projects/mutations_detection_temp/detection_model_22936725/checkpoint-19056',
                  # '/sci/backup/morani/lab/Projects/mutations_detection_temp/detection_model_22936684/checkpoint-6352',
                  #  '/sci/backup/morani/lab/Projects/mutations_detection_temp/detection_model_22936322/checkpoint-12704'
                   ]
    run_detection_on_test_dataset(model_paths, test_fasta_mutated, test_fasta_true, ignore_in_labels)
