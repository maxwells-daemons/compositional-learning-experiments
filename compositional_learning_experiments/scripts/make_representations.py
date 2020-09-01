#!/usr/bin/env python

"""
A script to write out a model's representations of a dataset.

NOTE: the current implementation is rather brittle and relies on some hyperparameters
stored in the config, as well as hardcoded behavior around various styles of data
loading. Ideally this will change at some point.
"""

import os
import pickle
import sys

import numpy as np
import omegaconf
import sympy
import torch

from compositional_learning_experiments import models, data


def main(ckpt_path):
    device = torch.device("cuda")  # TODO: support inference with other devices

    base_path = os.path.abspath(os.path.join(ckpt_path, "..", ".."))
    cfg_path = os.path.join(base_path, "hparams.yaml")
    out_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(base_path, f"representations-{out_name}.pkl")

    config = omegaconf.OmegaConf.load(cfg_path)
    model_class = models.loading.model_class(config.model_name)
    model = model_class.load_from_checkpoint(ckpt_path).to(device).eval()
    model.freeze()

    def get_reps_and_labels(path, depths):
        print(f"== On dataset: {path} ==")

        results = {}
        for depth in depths:
            ds = data.load_dataset(path, [depth], model.data_format)
            loader = model.make_dataloader(ds, train=False)
            reps = np.zeros((2 * len(ds), config.d_model))
            labels = []
            sympy_labels = []

            for i, batch in enumerate(loader):
                if i % 1000 == 0 and i:
                    print(i)

                # NOTE: hardcoded behavior for different data formats
                if model.data_format == "tree":
                    tree = batch[0]
                    _, left_rep, right_rep = model(tree)
                    reps[2 * i, :] = left_rep.detach().cpu().numpy()
                    reps[2 * i + 1, :] = right_rep.detach().cpu().numpy()

                    labels.append(str(tree.left))
                    labels.append(str(tree.right))
                    sympy_labels.append(str(sympy.simplify(tree.left.to_sympy())))
                    sympy_labels.append(str(sympy.simplify(tree.right.to_sympy())))
                else:
                    _, left_reps, right_reps = model(batch)
                    start_idx = 2 * batch.batch_size * i
                    end_idx_left = start_idx + batch.batch_size
                    end_idx_right = end_idx_left + batch.batch_size

                    reps[start_idx:end_idx_left, :] = left_reps.detach().cpu().numpy()
                    reps[end_idx_left:end_idx_right, :] = (
                        right_reps.detach().cpu().numpy()
                    )

                    labels.extend(model.text_field.reverse(batch.left))
                    labels.extend(model.text_field.reverse(batch.right))
                    # TODO: make sympy labels available in this case

            results[depth] = {
                "representations": reps,
                "labels": labels,
                "sympy_labels": sympy_labels,
            }

        return results

    stored = {
        "train": get_reps_and_labels(config.train_path, config.train_depths),
        "val": get_reps_and_labels(config.val_path, config.val_depths),
        "test": get_reps_and_labels(config.test_path, config.test_depths),
    }

    with open(out_path, "wb") as f:
        pickle.dump(stored, f)

    print(f"== Done! ==\nWrote results to {out_path}")


if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    main(ckpt_path)
