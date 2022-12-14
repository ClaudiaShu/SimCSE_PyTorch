import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel, RobertaConfig
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class SimcseModelUnsup(nn.Module):
    """Simcse unsupercised"""

    def __init__(self, *args, **kwargs):
        super(SimcseModelUnsup, self).__init__()
        self.args = kwargs['args']

        pretrained_model = self.args.pretrained_model
        if pretrained_model is None:
            config = BertConfig()
            self.model = BertModel(config)
        elif self.args.arch == 'roberta':
            config = RobertaConfig.from_pretrained(pretrained_model)
            config.attention_probs_dropout_prob = self.args.dropout
            config.hidden_dropout_prob = self.args.dropout
            self.model = AutoModel.from_pretrained(pretrained_model, config=config)
            if self.args.do_mlm:
                self.lm_head = RobertaLMHead(config)
        elif self.args.arch == 'bert':
            config = BertConfig.from_pretrained(pretrained_model)
            config.attention_probs_dropout_prob = self.args.dropout
            config.hidden_dropout_prob = self.args.dropout
            self.model = AutoModel.from_pretrained(pretrained_model, config=config)
            if self.args.do_mlm:
                self.lm_head = BertLMPredictionHead(config)
        else:
            raise ValueError("Unsupported pretrained model")

    def forward(self, input_ids, attention_mask, token_type_ids=None, output_hidden_states=True):
        if self.args.arch == 'roberta':
            out = self.model(input_ids, attention_mask, output_hidden_states=output_hidden_states)
        else:
            out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=output_hidden_states)

        if self.args.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.args.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        '''
            Pooling 
            last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 
            — Sequence of hidden-states at the output of the last layer of the model.
    
            pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) 
            — Last layer hidden-state of the first token of the sequence (classification token) after 
            further processing through the layers used for the auxiliary pretraining task. E.g. for 
            BERT-family of models, this returns the classification token after processing through a 
            linear layer and a tanh activation function. The linear layer weights are trained from the 
            next sentence prediction (classification) objective during pretraining.
    
            hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True 
            is passed or when config.output_hidden_states=True) 
            — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an 
            embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
    
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
            attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is 
            passed or when config.output_attentions=True) 
            — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, 
            sequence_length, sequence_length).
    
            Attentions weights after the attention softmax, used to compute the weighted average in 
            the self-attention heads.
        '''

        return out