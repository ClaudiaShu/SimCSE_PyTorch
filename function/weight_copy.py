from transformers import BertForSequenceClassification, BertConfig
jetfire = BertForSequenceClassification.from_pretrained('bert-base-cased')
config = BertConfig.from_pretrained('bert-base-cased')

optimus = BertForSequenceClassification(config)

parts = ['bert.embeddings.word_embeddings.weight'
,'bert.embeddings.position_embeddings.weight'
,'bert.embeddings.token_type_embeddings.weight'
,'bert.embeddings.LayerNorm.weight'
,'bert.embeddings.LayerNorm.bias']

def joltElectrify (jetfire, optimus, parts):
  target = dict(optimus.named_parameters())
  source = dict(jetfire.named_parameters())

  for part in parts:
    target[part].data.copy_(source[part].data)

joltElectrify(jetfire, optimus, parts)