# encoding: utf-8


from transformers import BertConfig


class BertNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("model_dropout", 0.1)

class RobertaQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(RobertaQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("model_dropout", 0.1)
