from IPython.utils.py3compat import input
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, AlbertConfig
from transformers import RobertaConfig
from transformers import RobertaModel
"""
from transformers.modeling_albert import AlbertEmbeddings, AlbertTransformer
from transformers.modeling_albert import AlbertPreTrainedModel, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import RobertaLMHead
"""
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertTransformer, AlbertPreTrainedModel, ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.roberta.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaLMHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import gelu
import math
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import numpy as np


# %%
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # fix 住 Linear 层以外的
        # for p in self.parameters():
        #     p.requires_grad = False

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
                              2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# %%
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # fix 住 Linear 层以外的
        # for p in self.parameters():
        #     p.requires_grad = False

        # self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.view(
                -1,
                attention_mask.size(-1)) if attention_mask is not None else None
        else:
            attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # print(outputs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # pre_logits = self.pre_classifier(pooled_output)
        # logits = self.classifier(F.relu(pre_logits))

        logits = self.classifier(pooled_output)
        # print(logits)
        reshaped_logits = logits.view(-1, num_choices)
        # print(reshaped_logits.shape)

        outputs = (reshaped_logits,) + outputs[
                                       2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class RobertaForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    }
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(0.4)
        # self.loss_fct = FocalLoss(3)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits, pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                                                2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            # loss = self.loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


# %%
class RobertaForMultipleChoiceWithLM2(nn.Module):
    def __init__(self, tokenizer):
        super(RobertaForMultipleChoiceWithLM2, self).__init__()
        self.roberta_lm = RobertaForMaskedLM.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=RobertaConfig.from_pretrained('roberta-large'))
        self.roberta = RobertaForMultipleChoice.from_pretrained(
            'pre_weights/roberta-large_model.bin', config=RobertaConfig.from_pretrained('roberta-large'))
        self.tokenizer = tokenizer
        self.lamda = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        output1 = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               labels=labels, position_ids=position_ids, head_mask=head_mask)
        input_ids_tmp = attention_mask_tmp = token_type_ids_tmp = position_ids_tmp = head_mask_tmp = None
        if input_ids is not None:
            input_ids_tmp = input_ids.reshape(-1, input_ids.shape[-1])
        if attention_mask is not None:
            attention_mask_tmp = attention_mask.reshape(-1, attention_mask.shape[-1])
            # for i in range(attention_mask_tmp.shape[0]):
            #     for j in range(attention_mask_tmp.shape[1]):
            #         if input_ids_tmp[i][j] != self.tokenizer.sep_token_id:
            #             attention_mask_tmp[i][j] = 0
            #         else:
            #             attention_mask_tmp[i][j] = 0
            #             break
        if token_type_ids is not None:
            token_type_ids_tmp = token_type_ids.reshape(-1, token_type_ids.shape[-1])
        if position_ids is not None:
            position_ids_tmp = position_ids.reshape(-1, position_ids.shape[-1])
        if head_mask is not None:
            head_mask_tmp = head_mask.reshape(-1, head_mask.shape[-1])

        output2 = self.roberta_lm(input_ids=input_ids_tmp, attention_mask=attention_mask_tmp,
                                  token_type_ids=token_type_ids_tmp,
                                  position_ids=position_ids_tmp, head_mask=head_mask_tmp,
                                  masked_lm_labels=input_ids_tmp)
        output2 = output2[0].reshape(-1, input_ids.shape[-1]).mean(dim=1).reshape(-1, input_ids.shape[-2])
        if labels is not None:
            loss2 = CrossEntropyLoss()(-output2, labels)
            output1 = (output1[0] + self.lamda * self.lamda * loss2,) + output1[1:]
        return output1


# %%
class RobertaForMultipleChoiceWithLM(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    }
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoiceWithLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.lamda 控制语言模型辅助程度
        self.lamda1 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.lamda2 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        if True:
            '''语言模型 loss'''
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            '''
            masked_lm_loss 是一个长度为 (batch_size * num_choices * max_seq_length,) 的 Tensor
            需要将其转换为 (batch_size, num_choices, max_seq_length)，再对每一个 (max_seq_length) 求平均
            即每个问题三个选项分别计算出 loss，lm_loss.shape = (batch_size, num_choices)
            '''
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size), input_ids.reshape(-1))
            lm_loss = masked_lm_loss.view_as(input_ids).mean(dim=2)

            '''
            在 lm_loss 的基础上做一个分类问题，lm_loss 较低的认为更正确，因此取 -lm_loss
            '''
            loss_fct.reduction = 'mean'
            lm_loss_for_classification = loss_fct(-lm_loss, labels)

        outputs = (reshaped_logits,
                   pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                               2:]  # add hidden states and attention if they are here
        # outputs = (reshaped_logits / (2.0 * self.lamda1 * self.lamda1) + \
        #            -lm_loss / (2.0 * self.lamda2 * self.lamda2),
        #            pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
        #                                                                        2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            # 这里平衡一下两个 loss
            loss = (1.0 / (2.0 * self.lamda1 * self.lamda1) * loss) + (
                    1.0 / (2.0 * self.lamda2 * self.lamda2) * lm_loss_for_classification) + torch.log(
                self.lamda1 * self.lamda2)

            # low = max(loss, lm_loss_for_classification) + 1e-7
            # loss = loss * loss / low + \
            #        lm_loss_for_classification * lm_loss_for_classification / low

            # loss = loss + self.lamda1 * lm_loss_for_classification
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    # load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 这里是自己修改的，为了支持 attention 矩阵
        extended_attention_mask = attention_mask.unsqueeze(1)
        if attention_mask.dim() != 3:
            extended_attention_mask = extended_attention_mask.unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
                                                     1:]  # add hidden_states and attentions if they are here
        return outputs


# %%
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "albert"

    def __init__(self, config):
        super(AlbertForMultipleChoice, self).__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None

        outputs = self.albert(input_ids=flat_input_ids,
                              position_ids=flat_position_ids,
                              token_type_ids=flat_token_type_ids,
                              attention_mask=flat_attention_mask,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits, pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                                                2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# %%
class GPT2ForMultipleChoice(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(GPT2ForMultipleChoice, self).__init__()
        self.gpt2 = GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name_or_path, config=config)

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        outputs = self.gpt2(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            mc_labels=labels)
        return outputs[0], outputs[2]  # mc loss, mc logits


# %%
class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        from torch_geometric.nn import GATConv
        # nn1 = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # nn2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.conv1 = GINConv(nn1)
        # self.conv2 = GINConv(nn2)
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        x = self.conv1(x, edge_index, edge_weight)
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight)
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)

        x = gelu(logits)
        x = self.fc1(x)

        outputs = (x.reshape(-1, data.num_graphs), logits,)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], data.y.reshape(-1, data.num_graphs).argmax(dim=1))
            outputs = (loss,) + outputs

        return outputs

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        from torch.nn import Parameter
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GAT_GCNII_only(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GAT_GCNII_only, self).__init__()
        self.gcnii_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        from torch_geometric.utils import to_dense_adj
        from torch_geometric.nn import GATConv

        for _ in range(nlayers):
            self.gcnii_convs.append(GraphConvolution(nhidden, nhidden, variant=variant))

        for _ in range(nlayers):
            self.gat_convs.append(GATConv(nhidden, nhidden))

        self.gat_integrate = GATConv(2*nhidden, nhidden)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        # self.features_layer = nn.Linear(nhidden, default_dimension)
        self.params1 = list(self.gcnii_convs.parameters())
        self.params2 = list(self.fcs.parameters())

        self.coef1 = nn.Parameter(torch.rand(1))
        self.coef2 = nn.Parameter(torch.rand(1))

        #self.coef1 = 0.01
        # self.coef2 = 0.99

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.fc_adaptive = nn.Linear(nclass,   1)
        self.fc1 = nn.Linear(nhidden, 1)
        self.to_dense_adj = to_dense_adj

    def integrate(self, gcnii_layer, gat_layer):
        integrate_layer = self.coef1 ** 2 * gcnii_layer + self.coef2 ** 2 * gat_layer + self.coef1 * self.coef2 * gcnii_layer * gat_layer
        return integrate_layer

    def GAT_integrate(self, gcnii_layer, gat_layer, edge_index):
        integrate_layer = self.gat_integrate(torch.cat((gcnii_layer, gat_layer), dim=1), edge_index)
        return integrate_layer
    def customized_integrate(self, gcnii_layer, gat_layer):
        integrate_layer = (1/self.coef1**2) * gcnii_layer + (1/self.coef2**2) * gat_layer +(1/ (self.coef1*self.coef2))*gcnii_layer*gat_layer
        return integrate_layer

    def harmonic_integrate(self, gcnii_layer, gat_layer):
        coef1 = gcnii_layer.mean()
        coef2 = gat_layer.mean()
        gcnii_coef = (coef1) / (coef2+coef1)
        gat_coef = (coef2) / (coef2 + coef1)
        integrate_layer = gcnii_coef * gcnii_layer + gat_coef * gat_layer
        return integrate_layer

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        adj = self.to_dense_adj(edge_index)
        adj_size = adj.size()

        adj = torch.reshape(adj, [adj_size[1], adj_size[2]])
        if x.size()[0] != adj.size()[0]:
            assert False
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for (i, gcnii_conv), (index, gat_conv) in zip(enumerate(self.gcnii_convs), enumerate(self.gat_convs)):
            gcnii_layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            gcnii_layer_inner = self.act_fn(gcnii_conv(gcnii_layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))

            gat_layer_inner = gat_conv(layer_inner, edge_index)

            layer_inner = self.integrate(gcnii_layer_inner, gat_layer_inner)

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        merged_features = torch.stack([layer_inner[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)
        logits = self.fc_adaptive(merged_features)
        logitsT = logits.T
        # soft = F.softmax(logitsT)
        # soft = F.softmax(logits, dim=1)
        outputs = (logits.reshape(-1, data.num_graphs), logitsT)

        # log_softmax = F.log_softmax(layer_inner, dim=1)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            # y = graph_features.y.reshape(-1, graph_features.num_graphs).argmax(dim=1)#something wrong here
            y = data.y.reshape(-1, data.num_graphs)[::, 0]
            loss = loss_fct(logits.T, y)
            outputs = (loss,) + outputs

        return outputs


class GAT_GCNII_latent_only(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, gamma, variant):
        super(GAT_GCNII_latent_only, self).__init__()
        self.gcnii_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        from torch_geometric.utils import to_dense_adj
        from torch_geometric.nn import GATConv

        for _ in range(nlayers):
            self.gcnii_convs.append(GraphConvolutionMPS(nhidden, nhidden, variant=variant))

        for _ in range(nlayers):
            self.gat_convs.append(GATConv(nhidden, nhidden))

        self.gat_integrate = GATConv(2*nhidden, nhidden)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nhidden))

        # self.features_layer = nn.Linear(nhidden, default_dimension)
        self.params1 = list(self.gcnii_convs.parameters())
        self.params2 = list(self.fcs.parameters())

        self.coef1 = nn.Parameter(torch.rand(1))
        self.coef2 = nn.Parameter(torch.rand(1))

        #self.coef1 = 0.01
        # self.coef2 = 0.99

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma

        self.fc_adaptive = nn.Linear(nhidden,   1)
        self.fc1 = nn.Linear(nhidden, 1)
        self.to_dense_adj = to_dense_adj

    def integrate(self, gcnii_layer, gat_layer):
        integrate_layer = self.coef1 ** 2 * gcnii_layer + self.coef2 ** 2 * gat_layer + self.coef1 * self.coef2 * gcnii_layer * gat_layer
        return integrate_layer

    def GAT_integrate(self, gcnii_layer, gat_layer, edge_index):
        integrate_layer = self.gat_integrate(torch.cat((gcnii_layer, gat_layer), dim=1), edge_index)
        return integrate_layer
    def customized_integrate(self, gcnii_layer, gat_layer):
        integrate_layer = (1/self.coef1**2) * gcnii_layer + (1/self.coef2**2) * gat_layer +(1/ (self.coef1*self.coef2))*gcnii_layer*gat_layer
        return integrate_layer

    def harmonic_integrate(self, gcnii_layer, gat_layer):
        coef1 = gcnii_layer.mean()
        coef2 = gat_layer.mean()
        gcnii_coef = (coef1) / (coef2+coef1)
        gat_coef = (coef2) / (coef2 + coef1)
        integrate_layer = gcnii_coef * gcnii_layer + gat_coef * gat_layer
        return integrate_layer

    def forward(self, input):
        # x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        graph_features, graph_dep_path, graph_ret_path = input
        x, edge_index = graph_features.x, graph_features.edge_index
        graph_adj = self.to_dense_adj(edge_index)
        graph_adj = graph_adj.view(-1, graph_adj.size()[-1])
        graph_dep_x, graph_dep_edge_index = graph_dep_path.x, graph_dep_path.edge_index
        graph_ret_x, graph_ret_edge_index = graph_ret_path.x, graph_ret_path.edge_index


        adj = self.to_dense_adj(edge_index)
        adj_size = adj.size()

        adj = torch.reshape(adj, [adj_size[1], adj_size[2]])
        if x.size()[0] != adj.size()[0]:
            assert False
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for (i, gcnii_conv), (index, gat_conv) in zip(enumerate(self.gcnii_convs), enumerate(self.gat_convs)):
            gcnii_layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            gcnii_layer_inner = self.act_fn(gcnii_conv(gcnii_layer_inner, adj, _layers[0], self.lamda, self.alpha, self.gamma, i + 1, graph_dep_x, graph_ret_x, graph_features))

            gat_layer_inner = gat_conv(layer_inner, edge_index)

            layer_inner = self.integrate(gcnii_layer_inner, gat_layer_inner)

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        merged_features = torch.stack([layer_inner[graph_features.batch == i][graph_features.pos[graph_features.batch == i]].mean(dim=0) for i in range(graph_features.num_graphs)], dim=0)
        logits = self.fc_adaptive(merged_features)
        logitsT = logits.T
        soft = F.softmax(logitsT)
        # soft = F.softmax(logits, dim=1)
        outputs = (logits.reshape(-1, graph_features.num_graphs), merged_features)

        # log_softmax = F.log_softmax(layer_inner, dim=1)

        if graph_features.y is not None:
            loss_fct = CrossEntropyLoss()
            # y = graph_features.y.reshape(-1, graph_features.num_graphs).argmax(dim=1)#something wrong here
            y = graph_features.y.reshape(-1, graph_features.num_graphs)[::, 0]
            loss = loss_fct(logitsT, y)
            outputs = (loss,) + outputs

        return outputs

class GCNII_only(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII_only, self).__init__()
        self.gcnii_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()
        for _ in range(nlayers):
            self.gcnii_convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        # self.features_layer = nn.Linear(nhidden, default_dimension)
        self.params1 = list(self.gcnii_convs.parameters())
        self.params2 = list(self.fcs.parameters())

        self.coef1 = 0.01
        self.coef2 = 0.99

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.fc_adaptive = nn.Linear(nclass,   1)
        from torch_geometric.utils import to_dense_adj
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.conv3 = GATConv(128, 128)
        self.conv4 = GATConv(128, 128)
        self.gat_convs.append(self.conv2)
        self.gat_convs.append(self.conv3) # self.gat_convs.append(self.conv4)
        self.fc1 = nn.Linear(128, 1)
        self.to_dense_adj = to_dense_adj

    def integrate(self, gcnii_layer, gat_layer):
        integrate_layer = self.coef1**2 * gcnii_layer + self.coef2**2 * gat_layer + self.coef1*self.coef2*gcnii_layer*gat_layer
        return integrate_layer

    def harmonic_integrate(self, gcnii_layer, gat_layer):
        coef1 = gcnii_layer.mean()
        coef2 = gat_layer.mean()
        gcnii_coef = (coef1) / (coef2+coef1)
        gat_coef = (coef2) / (coef2 + coef1)
        integrate_layer = gcnii_coef * gcnii_layer + gat_coef * gat_layer
        return integrate_layer

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        adj = self.to_dense_adj(edge_index)
        adj_size = adj.size()

        adj = torch.reshape(adj, [adj_size[1], adj_size[2]])
        if x.size()[0] != adj.size()[0]:
            assert False
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for (i, gcnii_conv), (index, gat_conv) in zip(enumerate(self.gcnii_convs), enumerate(self.gat_convs)):
            gcnii_layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            gcnii_layer_inner = self.act_fn(gcnii_conv(gcnii_layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))

            gat_layer_inner = gat_conv(layer_inner, edge_index)

            layer_inner = self.integrate(gcnii_layer_inner, gat_layer_inner)


        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        merged_features = torch.stack([layer_inner[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)
        logits = self.fc_adaptive(merged_features)
        logitsT = logits.T
        # soft = F.softmax(logitsT)
        # soft = F.softmax(logits, dim=1)
        outputs = (logits.reshape(-1, data.num_graphs), logitsT)

        # log_softmax = F.log_softmax(layer_inner, dim=1)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            # y = graph_features.y.reshape(-1, graph_features.num_graphs).argmax(dim=1)#something wrong here
            y = data.y.reshape(-1, data.num_graphs)[::, 0]
            loss = loss_fct(logits.T, y)
            outputs = (loss,) + outputs

        return outputs

class GCN_only(nn.Module):
    def __init__(self):
        super(GCN_only, self).__init__()
        from torch_geometric.nn import GATConv
        # nn1 = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # nn2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.conv1 = GINConv(nn1)
        # self.conv2 = GINConv(nn2)
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.conv3 = GATConv(128, 128)
        self.conv4 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        x = self.conv1(x, edge_index, edge_weight)
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits2 = self.conv2(x, edge_index, edge_weight)
        # logits3 = self.conv3(logits2, edge_index, edge_weight)
        # logits4 = self.conv4(logits3, edge_index, edge_weight)
        logits = logits2
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)

        x = gelu(logits)
        x = self.fc1(x)

        outputs = (x.reshape(-1, data.num_graphs), logits,)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            y = data.y.reshape(-1, data.num_graphs)[::, 0]
            loss = loss_fct(outputs[0], y)
            outputs = (loss,) + outputs

        return outputs

        """
        _layers = []
        x = graph_x
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, graph_adj, _layers[0], self.lamda, self.alpha, self.gamma, i + 1, graph_dep_x, graph_ret_x))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        # last_layer_inner = self.fcs[-1](layer_inner)
        last_layer_inner = self.features_layer(layer_inner)
        rescaled_layer = torch.stack([last_layer_inner[graph_features.batch == i][graph_features.pos[graph_features.batch == i]].mean(dim=0) for i in range(graph_features.num_graphs)], dim=0)
        logits = self.fc_adaptive(rescaled_layer)
        logitsT = logits.T
        soft = F.softmax(logitsT, dim=1)
        outputs = (logits.reshape(-1, graph_features.num_graphs), rescaled_layer)

        # log_softmax = F.log_softmax(layer_inner, dim=1)

        if graph_features.y is not None:
            loss_fct = CrossEntropyLoss()
            # y = graph_features.y.reshape(-1, graph_features.num_graphs).argmax(dim=1)#something wrong here
            y = graph_features.y.reshape(-1, graph_features.num_graphs)[::, 0]
            loss = loss_fct(soft, y)
            outputs = (loss,) + outputs

        return outputs
        """

class KGGCNNet(nn.Module):
    def __init__(self):
        super(KGGCNNet, self).__init__()
        from torch_geometric.nn import GATConv
        # nn1 = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # nn2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.conv1 = GINConv(nn1)
        # self.conv2 = GINConv(nn2)
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT

        x = self.conv1(x, edge_index, edge_weight)
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight)
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)

        x = gelu(logits)
        x = self.fc1(x)

        outputs = (x.reshape(-1, data.num_graphs), logits,)

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], data.y.reshape(-1, data.num_graphs).argmax(dim=1))
            outputs = (loss,) + outputs

        return outputs


class SOTA_goal_model(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2

        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)

        from utils.attentionUtils import SelfAttention
        self.gcn = KGGCNNet()
        self.merge_fc1 = nn.Linear(roberta_config.hidden_size + 128, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + roberta_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j[1] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][1].shape).to(
            self.args['device'])
        attention_mask = torch.stack([j[2] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][2].shape).to(
            self.args['device'])
        token_type_ids = torch.stack([j[3] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][3].shape).to(self.args['device'])
        position_ids = torch.stack([j[4] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][4].shape).to(
            self.args['device'])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]

        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        graph_features = [i[1].to('cpu') for i in x]

        loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        roberta_logits = roberta_outputs[2]

        loss = loss + torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        del gcn_tmp_features, roberta_outputs  # 清理显存

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        merge_features = self.merge_fc1(
            torch.cat((roberta_logits, gcn_features), dim=2))
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(torch.cat((roberta_logits, merge_features), dim=2)).view(-1, num_choices)
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss + loss_fct(outputs[0], labels)  # merge loss

            outputs = (loss,) + outputs
        return outputs


class SOTA_goal_model_plm_only(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model_plm_only, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []

        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)

        from utils.attentionUtils import SelfAttention
        # self.gcn = GCNNet()
        self.merge_fc1 = nn.Linear(roberta_config.hidden_size, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + roberta_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j[1] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][1].shape).to(
            self.args['device'])
        attention_mask = torch.stack([j[2] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][2].shape).to(
            self.args['device'])
        token_type_ids = torch.stack([j[3] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][3].shape).to(self.args['device'])
        position_ids = torch.stack([j[4] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][4].shape).to(
            self.args['device'])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        # gcn_tmp_features = [self.gcn(i) for i in graph_features]
        # random argument algorithm to generate more samples
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        # graph_features = [i[1].to('cpu') for i in x]

        roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        roberta_logits = roberta_outputs[2]

        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = roberta_logits.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = roberta_logits.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        """
        def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)
        """
        loss_all = roberta_loss
        if self.args.get("with_scl"):
            loss_all = roberta_loss + loss_contrastive
        # here to add the supervised contrastive loss.

        # gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        # loss_all = loss_all + gcn_loss
        loss_all = loss_all
        # gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        # merge_features = self.merge_fc1(torch.cat((roberta_logits, gcn_features), dim=2))
        merge_features = self.merge_fc1(roberta_logits)
        # merge_features = roberta_logits
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(torch.cat((roberta_logits, merge_features), dim=2)).view(-1, num_choices)
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,
        gcn_loss = torch.tensor(0)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            fct_loss = loss_fct(outputs[0], labels)  # merge loss
            loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss, loss_contrastive)
            else:
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss)

        return outputs

class SOTA_goal_model_gnn_only(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model_gnn_only, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []
        """
        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        """
        # from utils.attentionUtils import SelfAttention
        self.gcn = GCN_only()
        # self.gcn = GCNII_only(300, 3, 128, 2, .3, .2, .2, .2)
        # self.merge_fc1 = nn.Linear(128, 512)
        # self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        # self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]
        # gcn_tmp_features = list(filter(lambda x: x is not None, gcn_tmp_features))

        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])
        gcn_logit = torch.stack([i[1] for i in gcn_tmp_features]).view(-1, num_choices) # [batch_size, 1, 128]
        # random argument algorithm to generate more samples
        """"
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)
        """
        # graph_features = [i[1].to('cpu') for i in x]

        # roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        # roberta_logits = roberta_outputs[2]
        gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss


        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = gcn_features.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = gcn_features.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        # loss_all = gcn_loss
        # here to add the supervised contrastive loss.

        # loss_all = loss_all
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        """
        merge_features = self.gcn_fc1(gcn_features)  # gcn fc1 128, 128 -> merge_features [32, 2, 128]
        merge_features = self.merge_fc1(merge_features)  # merge fc1 128, 512 -> merge_features[32, 2, 512]
        merge_features = self.attn(merge_features)[0]  # attn 512 512 -> merge_features[32, 2, 512]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(merge_features).view(-1, num_choices)  # fc3 [, 1]
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)
        outputs = merge_features,
        """
        roberta_loss = torch.tensor(0)
        if labels is not None:
            gcn_loss = gcn_loss
            # loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                loss_all = gcn_loss + loss_contrastive
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss, loss_contrastive)
            else:
                loss_all = gcn_loss
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss)

        return outputs


class SOTA_goal_model_gat_gcnii_only(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model_gat_gcnii_only, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []
        """
        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        """
        # from utils.attentionUtils import SelfAttention
        self.gcn = GAT_GCNII_latent_only(300, 3, 128, 2, .3, .2, .2, .95, False)
        # self.gcn = GCNII_only(300, 3, 128, 2, .3, .2, .2, .2)
        # self.merge_fc1 = nn.Linear(128, 512)
        # self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        # self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])

        # graph_features = [i[1].to(self.args['device']) for i in x]
        # labels = labels.to(self.args['device'])

        # graph_features = [i[1][0].to(self.args['device']) for i in x]
        if self.args.get("with_mps"):
            graph_features = [i[1][0] for i in x]
            # graph_dpt_features = [i[1][1].to(self.args['device']) for i in x]
            graph_dpt_features = [i[1][1] for i in x]
            # graph_ret_features = [i[1][2].to(self.args['device']) for i in x]
            graph_ret_features = [i[1][2] for i in x]
            graph_all_data = [[graph_features[_], graph_dpt_features[_], graph_ret_features[_]] for _ in range(len(graph_features))]
        else:
            graph_features = [i[1] for i in x]
            graph_all_data = graph_features


        gcn_tmp_features = [self.gcn(i) for i in graph_all_data]
        # gcn_tmp_features = list(filter(lambda x: x is not None, gcn_tmp_features))

        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])
        gcn_logit = torch.stack([i[1] for i in gcn_tmp_features]).view(-1, num_choices) # [batch_size, 1, 128]

        """"
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)
        """
        # graph_features = [i[1].to('cpu') for i in x]

        # roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        # roberta_logits = roberta_outputs[2]
        gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        del gcn_tmp_features
        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = gcn_features.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = gcn_features.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        # loss_all = gcn_loss
        # here to add the supervised contrastive loss.

        # loss_all = loss_all
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        """
        merge_features = self.gcn_fc1(gcn_features)  # gcn fc1 128, 128 -> merge_features [32, 2, 128]
        merge_features = self.merge_fc1(merge_features)  # merge fc1 128, 512 -> merge_features[32, 2, 512]
        merge_features = self.attn(merge_features)[0]  # attn 512 512 -> merge_features[32, 2, 512]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(merge_features).view(-1, num_choices)  # fc3 [, 1]
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)
        outputs = merge_features,
        """
        roberta_loss = torch.tensor(0)
        if labels is not None:
            gcn_loss = gcn_loss
            # loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                loss_all = gcn_loss + loss_contrastive
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss, loss_contrastive)
            else:
                loss_all = gcn_loss
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss)
        if torch.isnan(gcn_loss):
            flag = False
            assert flag
        return outputs


class SOTA_goal_model_gcnii_only(nn.Module):
    def __init__(self, args):
        super(SOTA_goal_model_gcnii_only, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []
        """
        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        """
        # from utils.attentionUtils import SelfAttention
        # self.gcn = GCN_only()
        self.gcn = GCNII_only(300, 3, 128, 2, .3, .2, .2, .2)
        # self.merge_fc1 = nn.Linear(128, 512)
        # self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        # self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]
        # gcn_tmp_features = list(filter(lambda x: x is not None, gcn_tmp_features))

        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])
        gcn_logit = torch.stack([i[1] for i in gcn_tmp_features]).view(-1, num_choices) # [batch_size, 1, 128]
        # random argument algorithm to generate more samples
        """"
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)
        """
        # graph_features = [i[1].to('cpu') for i in x]

        # roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        # roberta_logits = roberta_outputs[2]
        gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss


        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = gcn_features.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = gcn_features.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        # loss_all = gcn_loss
        # here to add the supervised contrastive loss.

        # loss_all = loss_all
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        """
        merge_features = self.gcn_fc1(gcn_features)  # gcn fc1 128, 128 -> merge_features [32, 2, 128]
        merge_features = self.merge_fc1(merge_features)  # merge fc1 128, 512 -> merge_features[32, 2, 512]
        merge_features = self.attn(merge_features)[0]  # attn 512 512 -> merge_features[32, 2, 512]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(merge_features).view(-1, num_choices)  # fc3 [, 1]
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)
        outputs = merge_features,
        """
        roberta_loss = torch.tensor(0)
        if labels is not None:
            gcn_loss = gcn_loss
            # loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                loss_all = gcn_loss + loss_contrastive
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss, loss_contrastive)
            else:
                loss_all = gcn_loss
                outputs = (loss_all, gcn_logit, roberta_loss, gcn_loss, gcn_loss)

        return outputs


class GraphConvolutionMPS(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolutionMPS, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.IN = nn.InstanceNorm1d(in_features, affine=True)
        self.graph1_inner_LN = LayerNormalization(1)
        self.graph1_extern_LN = LayerNormalization(1)
        self.graph2_inner_LN = LayerNormalization(1)
        self.graph2_extern_LN = LayerNormalization(1)
        self.residual = residual
        from torch.nn.parameter import Parameter
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, gamma, l, depart_adj, return_adj, graph_features):
        theta = math.log(lamda / l + 1)
        hi = torch.mm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            hi_max = hi.max()
            hi_min = hi.min()

            latent_matrix = torch.mm(torch.mm(hi, depart_adj.T), return_adj)
            latent_max = latent_matrix.max()
            latent_min = latent_matrix.min()
            ratio = (hi_max - hi_min) / (latent_max - latent_min)
            ratio = 0 if torch.isinf(ratio) else ratio
            support = ((gamma) * ((1 - alpha) * hi
                                 + alpha * h0) +
                       (1 - gamma) * latent_matrix * ratio)

            r = support
        output = (theta * torch.mm(support, self.weight)
                  + (1 - theta) * r)

        output = self.gcnii_group_norm(output, graph_features)

        if self.residual:
            output = output + input
        return output


    def gcnii_group_norm(self, input, graph_features):
        # merged_features = torch.stack([input[graph_features.batch == i][graph_features.pos[graph_features.batch == i]].mean(dim=0) for i in range(graph_features.num_graphs)], dim=0)
        # input[graph_features.batch == 0][graph_features.pos[graph_features.batch == 0]]

        graph1_inner = input[graph_features.batch == 0][graph_features.pos[graph_features.batch == 0]]
        graph1_extern = input[graph_features.batch == 0][~graph_features.pos[graph_features.batch == 0]]
        graph2_inner = input[graph_features.batch == 1][graph_features.pos[graph_features.batch == 1]]
        graph2_extern = input[graph_features.batch == 1][~graph_features.pos[graph_features.batch == 1]]

        graph1_inner_norm = self.graph1_inner_LN(graph1_inner)
        graph1_extern_norm = self.graph1_extern_LN(graph1_extern)
        graph2_inner_norm = self.graph2_inner_LN(graph2_inner)
        graph2_extern_norm = self.graph2_extern_LN(graph2_extern)

        input[graph_features.batch == 0][graph_features.pos[graph_features.batch == 0]] = graph1_inner_norm
        input[graph_features.batch == 0][~graph_features.pos[graph_features.batch == 0]] = graph1_extern_norm
        input[graph_features.batch == 1][graph_features.pos[graph_features.batch == 1]] = graph2_inner_norm
        input[graph_features.batch == 1][~graph_features.pos[graph_features.batch == 1]] = graph2_extern_norm

        return input

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class GCNII_MPS(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, gamma, variant, default_dimension=128):
        super(GCNII_MPS, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolutionMPS(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        self.features_layer = nn.Linear(nhidden, default_dimension)
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma

        self.fc_adaptive = nn.Linear(default_dimension,   1)
        from torch_geometric.utils import to_dense_adj
        self.to_dense_adj = to_dense_adj

    def forward(self, x):
        graph_features, graph_dep_path, graph_ret_path = x
        graph_x, graph_edge_index = graph_features.x, graph_features.edge_index
        graph_adj = self.to_dense_adj(graph_edge_index)
        graph_adj = graph_adj.view(-1, graph_adj.size()[-1])
        graph_dep_x, graph_dep_edge_index = graph_dep_path.x, graph_dep_path.edge_index
        graph_ret_x, graph_ret_edge_index = graph_ret_path.x, graph_ret_path.edge_index

        # adj = graph_features
        _layers = []
        x = graph_x
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, graph_adj, _layers[0], self.lamda, self.alpha, self.gamma, i + 1, graph_dep_x, graph_ret_x, graph_features))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        # last_layer_inner = self.fcs[-1](layer_inner)
        last_layer_inner = self.features_layer(layer_inner)
        rescaled_layer = torch.stack([last_layer_inner[graph_features.batch == i][graph_features.pos[graph_features.batch == i]].mean(dim=0) for i in range(graph_features.num_graphs)], dim=0)
        logits = self.fc_adaptive(rescaled_layer)
        logitsT = logits.T
        soft = F.softmax(logitsT, dim=1)
        outputs = (logits.reshape(-1, graph_features.num_graphs), rescaled_layer)

        # log_softmax = F.log_softmax(layer_inner, dim=1)

        if graph_features.y is not None:
            loss_fct = CrossEntropyLoss()
            # y = graph_features.y.reshape(-1, graph_features.num_graphs).argmax(dim=1)#something wrong here
            y = graph_features.y.reshape(-1, graph_features.num_graphs)[::, 0]
            loss = loss_fct(soft, y)
            outputs = (loss,) + outputs

        return outputs

        #require reture (a,b) a -> loss b -> features output
        """
        self.conv1 = GATConv(300, 128)
        self.conv2 = GATConv(128, 128)
        self.fc1 = nn.Linear(128, 1)
        
        x, edge_index, edge_weight = data.x, data.edge_index, None  # for GAT
        x = self.conv1(x, edge_index, edge_weight) x from [n, 300] -> [n, 128]
        x = gelu(x)
        x = F.dropout(x, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight) logits -> [n, 128]
        logits = torch.stack(
            [logits[data.batch == i][data.pos[data.batch == i]].mean(dim=0) for i in range(data.num_graphs)], dim=0)
            

        x = gelu(logits)
        x = self.fc1(x) x from [n, 128] -> [n, 1]

        outputs = (x.reshape(-1, data.num_graphs), logits,) x from [n, 1] -> [n/2, 2] 

        if data.y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], data.y.reshape(-1, data.num_graphs).argmax(dim=1))
            outputs = (loss,) + outputs
"""


class SCL_SOTA_goal_model_GNN_PLM_GCNII(nn.Module):
    def __init__(self, args):
        super(SCL_SOTA_goal_model_GNN_PLM_GCNII, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []

        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)

        from utils.attentionUtils import SelfAttention
        # self.gcn = GCNNet()
        # self.gcnii = GAT_GCNII_latent_only(300, 3, 128, 2, .3, .2, .2, .95, False) # original
        self.gcnii = GAT_GCNII_latent_only(300, 3, 256, 2, .3, .2, .2, .95, False)

        # self.merge_fc1 = nn.Linear(roberta_config.hidden_size + 256, 512) # original 128
        self.merge_fc1 = nn.Linear(roberta_config.hidden_size + 256, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + roberta_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j[1] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][1].shape)#.to(self.args['device'])
        attention_mask = torch.stack([j[2] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][2].shape)#.to(self.args['device'])
        # token_type_ids = torch.stack([j[3] for i in semantic_features for j in i], dim=0).reshape(
            #(-1, num_choices,) + semantic_features[0][0][3].shape).to(self.args['device'])
        position_ids = torch.stack([j[4] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][4].shape)#.to(self.args['device'])

        # graph_features = [i[1][0].to(self.args['device']) for i in x]
        graph_features = [i[1][0] for i in x]
        # graph_dpt_features = [i[1][1].to(self.args['device']) for i in x]
        graph_dpt_features = [i[1][1] for i in x]
        # graph_ret_features = [i[1][2].to(self.args['device']) for i in x]
        graph_ret_features = [i[1][2] for i in x]
        graph_all_data = [[graph_features[_], graph_dpt_features[_], graph_ret_features[_]] for _ in range(len(graph_features))]

        # labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcnii(i) for i in graph_all_data]
        # random argument algorithm to generate more samples
        # del graph_features, graph_dpt_features, graph_ret_features, graph_all_data
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        # graph_features = [i[1].to('cpu') for i in x]

        roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        roberta_logits = roberta_outputs[2]

        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = roberta_logits.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = roberta_logits.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        """
        def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)
        """
        loss_all = roberta_loss
        if self.args.get("with_scl"):
            loss_all = roberta_loss + loss_contrastive
        # here to add the supervised contrastive loss.

        gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        loss_all = loss_all + gcn_loss
        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        merge_features = self.merge_fc1(torch.cat((roberta_logits, gcn_features), dim=2)) #fc1 roberta logit 8, 2, 1024 gcn 8, 2
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(torch.cat((roberta_logits, merge_features), dim=2)).view(-1, num_choices)
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            soft_outputs = F.softmax(outputs[0], dim=1)
            fct_loss = loss_fct(soft_outputs, labels)  # merge loss
            loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss, loss_contrastive)
            else:
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss)

        return outputs


# %%
class SCL_SOTA_goal_model(nn.Module):
    def __init__(self, args):
        super(SCL_SOTA_goal_model, self).__init__()
        self.args = args
        # roberta_config = AlbertConfig.from_pretrained('albert-base-v2')
        # self.roberta = AlbertForMultipleChoice.from_pretrained(
        #     'pre_weights/albert-base-v2-pytorch_model.bin', config=roberta_config)
        roberta_config = RobertaConfig.from_pretrained('roberta-large')
        roberta_config.attention_probs_dropout_prob = 0.2
        roberta_config.hidden_dropout_prob = 0.2
        history_grad_all = []
        history_grad_roberta = []
        history_grad_gcn = []
        history_grad_scl = []

        if args.get('with_lm'):
            self.roberta = RobertaForMultipleChoiceWithLM.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)
        else:
            self.roberta = RobertaForMultipleChoice.from_pretrained(
                'pre_weights/roberta-large_model.bin', config=roberta_config)

        from utils.attentionUtils import SelfAttention
        self.gcn = GCNNet()
        self.merge_fc1 = nn.Linear(roberta_config.hidden_size + 128, 512)
        self.attn = SelfAttention(512, 8)
        # self.roberta_fc1 = nn.Linear(roberta_config.hidden_size, 128)  # 将 roberta vector 降维到与 gcn 相同
        # self.gcn_fc1 = nn.Linear(128, 128)  # 同上
        self.fc3 = nn.Linear(512 + roberta_config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, labels=None):
        semantic_features = [i[0] for i in x]
        num_choices = len(semantic_features[0])
        input_ids = torch.stack([j[1] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][1].shape).to(
            self.args['device'])
        attention_mask = torch.stack([j[2] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][2].shape).to(
            self.args['device'])
        token_type_ids = torch.stack([j[3] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][3].shape).to(self.args['device'])
        position_ids = torch.stack([j[4] for i in semantic_features for j in i], dim=0).reshape(
            (-1, num_choices,) + semantic_features[0][0][4].shape).to(
            self.args['device'])

        graph_features = [i[1].to(self.args['device']) for i in x]
        labels = labels.to(self.args['device'])

        gcn_tmp_features = [self.gcn(i) for i in graph_features]
        # random argument algorithm to generate more samples
        roberta_outputs = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       # token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

        # graph_features = [i[1].to('cpu') for i in x]

        roberta_loss = roberta_outputs[0]  # roberta loss
        # roberta reshaped_logits
        roberta_logits = roberta_outputs[2]

        if self.args.get("with_scl"):
            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            mask_all = None
            # data argument tech, make the similar samples closer
            hidden_size = roberta_logits.size()
            hidden_batch_size = hidden_size[0]
            hidden_category_size = hidden_size[1]
            hidden_feature_size = hidden_size[2]
            contrastive_loss_temp = .05
            contrastive_loss_coef = 1
            # flatten_all_hidden = torch.nn.Flatten(roberta_logits)
            flatten_all_hidden = roberta_logits.view([hidden_category_size, hidden_batch_size, hidden_feature_size])

            pos_flatten = flatten_all_hidden[0, ::, ::]
            neg_flatten = flatten_all_hidden[1, ::, ::]

            reorder_flatten_all_hidden = torch.cat([pos_flatten, neg_flatten], 0).T
            tmp_mask = torch.zeros(reorder_flatten_all_hidden.T.size()[0], dtype=torch.int8)

            mask_true = tmp_mask.detach().clone()
            mask_true[0:pos_flatten.size()[0]] = 1
            mask_false = tmp_mask.detach().clone()
            mask_false[-neg_flatten.size()[0]::] = 1
            mask = mask_true | mask_false
            # for single gpu single batch setting the mask is equal to mask_all
            mask_all = mask.clone().detach() if mask_all is None else torch.cat((mask_all, mask), dim=0)

            transpose_reorder_flatten_all_hidden = reorder_flatten_all_hidden.T
            cos_logitss = cos_fn(reorder_flatten_all_hidden.unsqueeze(0),
                                 transpose_reorder_flatten_all_hidden.unsqueeze(-1))
            """
            flatten_all_hidden = flatten_all_hidden.view(hidden_batch_size * hidden_category_size, hidden_feature_size)
            pos_logitss = cos_fn(pos_flatten, flatten_all_hidden)
            _logitss = cos_fn(pos_flatten, flatten_all_hidden)  # (B * C, num_gpus * B * C)
            """
            _logitss = cos_logitss / contrastive_loss_temp
            _logitss_eye = torch.eye(_logitss.size()[0]).to(self.args['device'])
            logitss = _logitss * (1 - _logitss_eye) + _logitss_eye * -1e9

            contrastive_losses = []
            for i in range(hidden_batch_size * hidden_category_size):
                logits = logitss[i, :].clone()  # (num_gpus * B * C)
                logprobs = torch.nn.functional.log_softmax(logits, dim=0)  # (num_gpus * B * C)
                if mask_true[i] == 1:
                    m = mask_true.detach().clone()
                    m[i] = 0
                    positive_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-positive_logprob)
                elif mask_false[i] == 1:
                    m = mask_false.detach().clone()
                    m[i] = 0
                    negative_logprob = torch.logsumexp(logprobs[m == 1], dim=0)
                    contrastive_losses.append(-negative_logprob)

            loss_contrastive = torch.stack(contrastive_losses).mean()  # ()
            weight = mask.sum().float().mean().item() / mask_all.sum().float().mean().item()
            loss_contrastive = loss_contrastive * weight
            loss_contrastive = loss_contrastive * contrastive_loss_coef

        """
        def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss * self.alpha)
        """
        loss_all = roberta_loss
        if self.args.get("with_scl"):
            loss_all = roberta_loss + loss_contrastive
        # here to add the supervised contrastive loss.

        gcn_loss = torch.stack([i[0] for i in gcn_tmp_features]).mean()  # + gcn loss
        loss_all = loss_all + gcn_loss
        gcn_features = torch.stack([i[2] for i in gcn_tmp_features])  # [4, 3, 64]
        # del gcn_tmp_features, roberta_outputs

        # print(roberta_logits.shape)
        # print(gcn_features.shape)
        merge_features = self.merge_fc1(torch.cat((roberta_logits, gcn_features), dim=2))
        merge_features = self.attn(merge_features)[0]

        # roberta_logits = self.roberta_fc1(roberta_logits)
        # gcn_features = self.gcn_fc1(gcn_features)
        # merge_features = roberta_logits + gcn_features

        # roberta_logits 最后是 tanH 算出来的，这里用 gelu 好不好
        # merge_features = nn.Tanh()(merge_features)
        merge_features = gelu(merge_features)
        merge_features = self.dropout(merge_features)
        merge_features = self.fc3(torch.cat((roberta_logits, merge_features), dim=2)).view(-1, num_choices)
        # merge_features = (self.fc3(merge_features) + self.fc3(roberta_logits)).view(-1, num_choices)

        outputs = merge_features,

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            fct_loss = loss_fct(outputs[0], labels)  # merge loss
            loss_all = loss_all + fct_loss
            if self.args.get("with_scl"):
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss, loss_contrastive)
            else:
                outputs = (loss_all, merge_features, roberta_loss, gcn_loss, fct_loss)

        return outputs


# %%
if __name__ == '__main__':
    from transformers import *

    # net = RobertaForMultipleChoiceWithLM2()

    import numpy as np
    from bidict import bidict
    from collections import defaultdict
    from utils.GraphUtils import GraphUtils
    from utils.getGraphUtils import get_datas, get_data_from_task_2, load_graph_pickle, merge_graph_by_downgrade

    data = np.array(get_datas(
        get_data_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv'),
        get_data_from_task_2(
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
            './SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv')
    ))
    graph = GraphUtils()
    graph.init()
    graph.merge_graph_by_downgrade()

    words_to_id = bidict()  # 将一个词映射为 id
    words_encode_idx = 0  # 实现上述两种操作的 idx
    conceptnet_numberbatch_en = dict()
    mp = defaultdict(set)

    mp_all, node_id_to_label_all, _, _ = load_graph_pickle('pre_weights/res.pickle')
    mp_all, node_id_to_label_all = merge_graph_by_downgrade(mp_all, node_id_to_label_all)
    # x, edge_index, edge_weight = encode_index(mp)
