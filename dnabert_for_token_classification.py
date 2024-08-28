from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
from sklearn.utils import class_weight
import numpy as np


class BertForTokenClassification(BertPreTrainedModel):
    """Bert Model transformer with a token classification head.
    This head is a linear layer on top of the hidden-states output.
    """

    def __init__(self, config, num_labels=None, weight_classes=False):
        super().__init__(config)
        if num_labels is not None:
            self.num_labels = num_labels
        else:
            self.num_labels = config.num_labels
        self.config = config
        self.weight_classes = weight_classes
        self.bert = BertModel(config)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        # Labels for computing the token classification loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.weight_classes:
                # Compute weighted cross-entropy loss
                unique_labels_with_counts = torch.unique(labels.view(-1), return_counts=True)
                labels_unique_np = unique_labels_with_counts[0].cpu().detach().numpy()
                counts = unique_labels_with_counts[1].cpu().detach().numpy()
                sum_without_padding = counts[labels_unique_np != 100].sum()
                if self.weight_classes=='inverse':
                    weights_dict = {label: (sum_without_padding / count) for count, label in
                                    zip(counts, labels_unique_np) if label != -100}
                elif self.weight_classes=='effective_num_of_samples':
                    beta = 0.95
                    weights_dict = {label: ((1-beta) / (1-beta**count)) for count, label in
                                    zip(counts, labels_unique_np) if label != -100}
                else:
                    raise ValueError(f"Invalid class balancing technique: {self.weight_classes}")
                class_weights = class_weight.compute_class_weight(weights_dict, classes=np.array(list(weights_dict)),
                                                                  y=[max(x, 0) for x in
                                                                     labels.cpu().detach().numpy().flatten()])
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(
                -1))  # TODO do we want to keep the default mean reduction? what about masking the padding?

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
