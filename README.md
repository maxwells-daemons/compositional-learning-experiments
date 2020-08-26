# compositional-learning-experiments
Experiments with improving the ability of RNNs and Transformers to systematically generalize.

Work conducted with the lab group of [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).

NOTE: this is research code, and as such will not have as clean a history, organization, or documentation as project code.
A later release might improve on that situation.

## Running training

`poetry run python compositional_learning_experiments/scripts/run_experiment.py`

## Configuration

This project is configured with [Hydra](https://hydra.cc/), and managed with `compositional_learning_experiments/config/config.yaml` or command-line equivalents.
The default is:
```
poetry run python compositional_learning_experiments/scripts/run_experiment.py  --help
run_experiment is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

data: equation_verification_40k, equation_verification_small, equation_verification_tiny
model: siamese_lstm, siamese_transformer, tree_rnn, tree_transformer


== Config ==
Override anything in the config (foo.bar=value)

data:
  test_depths:
  - 3
  - 4
  test_path: recursiveMemNet/data/40k_train.json
  train_depths:
  - 1
  - 2
  train_path: recursiveMemNet/data/40k_train.json
  val_depths:
  - 1
  - 2
  val_path: recursiveMemNet/data/40k_val_shallow.json
model:
  d_model: 32
  dropout: 0.0
  similarity_metric: symmetric_bilinear
model_meta:
  name: TreeRNN
trainer:
  gpus: 1
  max_epochs: 5
training:
  batch_size: 1
  learning_rate: 0.001
```


## References
Lake, B. M. and Baroni, M. (2018). [Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks.](https://arxiv.org/abs/1711.00350) Proceedings of ICML 2018.

Arabshahi, F., Lu, Z., Singh, S., &amp; Anandkumar, A. (2019). [Tree Stack Memory Units](https://arxiv.org/abs/1911.01545). ArXiv Preprint.
