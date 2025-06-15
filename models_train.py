from transformers.modeling_outputs import TokenClassifierOutput
import transformers
import torch.nn as nn
import torch

class ModelForTetraTagging(transformers.GPT2ForTokenClassification):
  """
  Class for GPT2-Tetratagger.
  """
  def __init__(self, config):
    super().__init__(config)
    self.num_leaf_labels = config.task_specific_params['num_leaf_labels']
    self.num_internal_labels = config.task_specific_params['num_internal_labels']

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
  ):
    outputs = super().forward(
        input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    if labels is not None:
      logits = outputs.logits
      # Split logits into internal and leaf.
      internal_logits, leaf_logits = torch.split(
        logits, [self.num_internal_labels, self.num_leaf_labels], dim=-1)
      internal_labels = (labels // (self.num_leaf_labels + 1)) - 1
      leaf_labels = (labels % (self.num_leaf_labels + 1)) - 1

      # Mask inactive positions.
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_internal_labels = torch.where(active_loss,  internal_labels.view(-1),  torch.full_like(internal_labels.view(-1), -1))
        active_leaf_labels = torch.where(active_loss,  leaf_labels.view(-1),  torch.full_like(leaf_labels.view(-1), -1))
        active_internal_logits = internal_logits.view(-1, self.num_internal_labels)
        active_leaf_logits = leaf_logits.view(-1, self.num_leaf_labels)
      else:
        active_internal_labels = internal_labels.view(-1)
        active_leaf_labels = leaf_labels.view(-1)
        active_internal_logits = internal_logits.view(-1, self.num_internal_labels)
        active_leaf_logits = leaf_logits.view(-1, self.num_leaf_labels)

      # Compute loss.
      loss = (
          nn.CrossEntropyLoss(ignore_index=-1)(active_internal_logits, active_internal_labels) +
          nn.CrossEntropyLoss(ignore_index=-1)(active_leaf_logits, active_leaf_labels))
      outputs = TokenClassifierOutput(loss=loss, logits=logits)

    return outputs  

class ModelForTetraTaggingLlama(transformers.LlamaForTokenClassification):
  """
  Class for Llama-Tetratagger.
  """
  def __init__(self, config):
    super().__init__(config)
    self.num_leaf_labels = config.task_specific_params['num_leaf_labels']
    self.num_internal_labels = config.task_specific_params['num_internal_labels']

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=True,
  ):
    outputs = super().forward(
        input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    if labels is not None:
      logits = outputs.logits
      internal_logits, leaf_logits = torch.split(
        logits, [self.num_internal_labels, self.num_leaf_labels], dim=-1)
      internal_labels = (labels // (self.num_leaf_labels + 1)) - 1
      leaf_labels = (labels % (self.num_leaf_labels + 1)) - 1

      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_internal_labels = torch.where(active_loss,  internal_labels.view(-1),  torch.full_like(internal_labels.view(-1), -1))
        active_leaf_labels = torch.where(active_loss,  leaf_labels.view(-1),  torch.full_like(leaf_labels.view(-1), -1))
        active_internal_logits = internal_logits.view(-1, self.num_internal_labels)
        active_leaf_logits = leaf_logits.view(-1, self.num_leaf_labels)
      else:
        active_internal_labels = internal_labels.view(-1)
        active_leaf_labels = leaf_labels.view(-1)
        active_internal_logits = internal_logits.view(-1, self.num_internal_labels)
        active_leaf_logits = leaf_logits.view(-1, self.num_leaf_labels)

      loss = (
          nn.CrossEntropyLoss(ignore_index=-1)(active_internal_logits, active_internal_labels) +
          nn.CrossEntropyLoss(ignore_index=-1)(active_leaf_logits, active_leaf_labels))
      outputs = TokenClassifierOutput(loss=loss, logits=logits)

    return outputs
