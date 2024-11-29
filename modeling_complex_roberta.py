import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
# from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_utils import create_position_ids_from_input_ids
from modeling_complex_bert import ComplexBertEmbeddings, ComplexBertLayer, ComplexBertModel, ComplexBertPreTrainedModel, ComplexBertLayerNorm, gelu, complex_gelu, ComplexLinear, ComplexDropout, ComplexCrossEntropyLoss

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
	"roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
	"roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
	"roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
	"distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
	"roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
	"roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
	"The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
	ROBERTA_START_DOCSTRING,
)
class ComplexRobertaModel(ComplexBertModel):
	"""
	This class overrides :class:`~transformers.BertModel`. Please check the
	superclass for the appropriate documentation alongside usage examples.
	"""
	
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"
	
	def __init__(self, config):
		super().__init__(config)
		
		self.embeddings = ComplexRobertaEmbeddings(config)
		self.init_weights()
	
	def get_input_embeddings(self):
		return self.embeddings.word_embeddings
	
	def set_input_embeddings(self, value):
		self.embeddings.word_embeddings = value


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class ComplexRobertaForMaskedLM(ComplexBertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"
	
	def __init__(self, config):
		super().__init__(config)
		
		self.roberta = ComplexRobertaModel(config)
		self.lm_head = ComplexRobertaLMHead(config)
		
		self.init_weights()
	
	def get_output_embeddings(self):
		return self.lm_head.decoder
	
	@add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
			self,
			input_ids=None,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			masked_lm_labels=None,
	):
		r"""
		masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Labels for computing the masked language modeling loss.
			Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
			Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
			in ``[0, ..., config.vocab_size]``

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
		masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
			Masked language modeling loss.
		prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.

	Examples::

		from transformers import RobertaTokenizer, RobertaForMaskedLM
		import torch

		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		model = RobertaForMaskedLM.from_pretrained('roberta-base')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, masked_lm_labels=input_ids)
		loss, prediction_scores = outputs[:2]

		"""
		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		sequence_output = outputs[0]
		prediction_scores = self.lm_head(sequence_output)
		
		outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
		
		if masked_lm_labels is not None:
			loss_fct = ComplexCrossEntropyLoss()
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
			outputs = (masked_lm_loss,) + outputs
		
		return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

class ComplexRobertaLMHead(nn.Module):
	"""Roberta Head for masked language modeling."""
	
	def __init__(self, config):
		super().__init__()
		self.dense = ComplexLinear(config.hidden_size, config.hidden_size)
		self.layer_norm = ComplexBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		
		self.decoder = ComplexLinear(config.hidden_size, config.vocab_size, bias=False)
		self.bias = nn.Parameter(torch.zeros(config.vocab_size))
		
		# Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
		self.decoder.bias = self.bias
	
	def forward(self, features, **kwargs):
		x = self.dense(features)
		x = complex_gelu(x)
		x = self.layer_norm(x)
		
		# project back to size of vocabulary with bias
		x = self.decoder(x)
		
		return x


@add_start_docstrings(
	"""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
	on top of the pooled output) e.g. for GLUE tasks. """,
	ROBERTA_START_DOCSTRING,
)

class ComplexRobertaEmbeddings(ComplexBertEmbeddings):
	"""
	Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
	"""
	
	def __init__(self, config):
		super().__init__(config)
		self.padding_idx = 1
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
		self.position_embeddings = nn.Embedding(
			config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
		)
	
	def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
		if position_ids is None:
			if input_ids is not None:
				# Create the position ids from the input token ids. Any padded tokens remain padded.
				position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
			else:
				position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
		
		return super().forward(
			input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
		)
	
	def create_position_ids_from_inputs_embeds(self, inputs_embeds):
		""" We are provided embeddings directly. We cannot infer which are padded so just generate
		sequential position ids.

		:param torch.Tensor inputs_embeds:
		:return torch.Tensor:
		"""
		input_shape = inputs_embeds.size()[:-1]
		sequence_length = input_shape[1]
		
		position_ids = torch.arange(
			self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
		)
		return position_ids.unsqueeze(0).expand(input_shape)
