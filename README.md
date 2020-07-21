# compositional-learning-experiments
Experiments with improving the ability of RNNs and Transformers to systematically generalize.

Work conducted with the lab group of [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).

NOTE: this is research code, and as such will not have as clean a history, organization, or documentation as project code.
A later release might improve on that situation.

## Running training

`poetry run python compositional_learning_experiments/train_scan.py`

## Configuration

This project is configured with [Hydra](https://hydra.cc/), and managed with `compositional_learning_experiments/config/config.yaml` or command-line equivalents.
The default is:
```
poetry run python compositional_learning_experiments/train_scan.py --help
train_scan is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

model: transformer
split: addprim, filler, length, simple, template


== Config ==
Override anything in the config (foo.bar=value)

model:
  d_model: 128
  dropout: 0.0
  name: transformer
  nhead: 16
  num_decoder_layers: 2
  num_encoder_layers: 2
split:
  train: SCAN/length_split/tasks_train_length.txt
  val: SCAN/length_split/tasks_test_length.txt
trainer:
  gpus: 1
  max_epochs: 100
training:
  batch_size: 256
  learning_rate: 0.001
```


## References
Lake, B. M. and Baroni, M. (2018). [Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks.](https://arxiv.org/abs/1711.00350) Proceedings of ICML 2018.
