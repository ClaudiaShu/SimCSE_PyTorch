from torch import nn
from transformers import AutoModel, BertConfig


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, *args, **kwargs):
        super(SimcseModel, self).__init__()
        self.args = kwargs['args']
        self.pretrained_model = self.args.pretrained_model

        config = BertConfig.from_pretrained(self.pretrained_model)
        config.attention_probs_dropout_prob = self.args.dropout
        config.hidden_dropout_prob = self.args.dropout
        self.model = AutoModel.from_pretrained(self.pretrained_model, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return out