# SpanNER: Named Entity Re-/Recognition as Span Prediction
Two roles of span prediction models (boxes in blue): 
* as a base NER system 
* as a system combiner.

![show fig](https://github.com/anonymous4nlp/anonymous4nlp.github.io/raw/master/img/3selfdiag-flairelmo.png)

## Requirements
-  `python3`
- `torch==1.6`
- `pytorch-lightning==0.9.0`
- `tokenizers`
- `transformers`
- `tensorflow==2.2.0`
- `allennlp==0.8.4`

## Run
`./run.sh`

The shell scripts include the `conll03_spanPred.sh`, and five parameters need to be defined.

- `use_pred_pruning`: `bool`, whether to use pruning decoder;
- `use_spanLen`: `bool`, whether to use the span length feature;
- `use_morphology`: `bool`, whether to use the morphology feature of span;
- `use_span_weight`: `bool`, whether to set different weights for positive and negative spans on the training process;
- `neg_span_weight`: `float`, the weight of the negative spans.
- `gpus`: set the GPU used during training.

## Datasets

The datasets utilized in our paper including:

- CoNLL-2003 (in this repository.)
- WNUT-2016 
- WNUT-2017
- CoNLL-2002-Spanish
- CoNLL-2002-Dutch
- OntoNotes 5.0 (The domains we utilized in the paper: WB, MZ, BC, BN.) (Yor can download from [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) )


