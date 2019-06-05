import logging
from collections import defaultdict
import time
import multiprocessing
import os
import argparse

import numpy as np
from flexp import flexp

from vsbd.dataset import DatasetReader
from vsbd.models import Vowpal, UniformPolicy


def parse_args():
    """
    Parse input arguments of the program.
    """
    parser = argparse.ArgumentParser(
        description="Vowpal experiment"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to a directory where results should be saved"
    )
    parser.add_argument(
        "--vowpal_path",
        type=str,
        required=True,
        help="Path to the vowpal binary"
    )
    parser.add_argument(
        "--num_test_positions",
        type=int,
        required=True,
        help="Number of positions to evaluate the models on (K in paper)"
    )
    parser.add_argument(
        "--num_train_positions",
        type=int,
        default=None,
        help="Number of positions from the beginning to train the models on. "
             "If None, train on all positions."
    )
    parser.add_argument(
        "--cb_types",
        nargs="+",
        type=str,
        help="Contextual bandit types for vowpal (dm ips dr)"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Do not train the models (combine with --load_model_dir)"
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="Load models (named model_{dm,ips,dr}.vw) from given directory"
    )
    parser.add_argument(
        "--save_vowpal_input",
        action="store_true",
        help="Save vowpal input for debug purposes"
    )
    parser.add_argument(
        "--save_propensities",
        action="store_true",
        help="Save propensities of trained model predictions on test set"
    )
    return parser.parse_args()


# shared chunk for multiprocessing
_chunk = {}


# multiprocessing helper function
def update_model(key):
    models[key].update(_chunk["chunk"])


args = parse_args()
num_test_positions = args.num_test_positions
num_train_positions = args.num_train_positions
test_positions = (
    list(range(num_test_positions))
    if num_test_positions else None
)
train_positions = (
    list(range(num_train_positions))
    if num_train_positions else None
)

train_days = ["201808{:02d}".format(i) for i in range(13, 32)]
test_days = ["201809{:02d}".format(i) for i in range(1, 22)]

flexp.setup(
    args.results_dir,
    "linear_pos_vowpal_num_train_pos_{}_num_test_pos_{}_cb_{}".format(
        num_train_positions,
        num_test_positions,
        "_".join(args.cb_types)
    ),
    with_date=True
)
flexp.describe(
    "save_all_input_vowpal {}, train on {}, test on {}, "
    "num_train_positions {}, num_test_positions {}"
    .format(
        "_".join(args.cb_types), train_days, test_days,
        num_train_positions, num_test_positions
    )
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
)

np.random.seed(25)

logging.info("Reading dataset")
train = DatasetReader(
    [
        os.path.join(args.dataset_dir, str(train_day))
        for train_day in train_days
    ],
    chunksize=100000,
    make_vowpal_train_input=True, use_positions=train_positions
).read()

test = DatasetReader(
    [
        os.path.join(args.dataset_dir, str(test_day))
        for test_day in test_days
    ],
    chunksize=100000,
    make_vowpal_test_input=True, use_positions=test_positions
).read()

num_actions = 21

models = {
    "uniform": UniformPolicy(num_actions=num_actions),
}
for cb_type in args.cb_types:
    model_filename = "model_{}.vw".format(cb_type)
    models["vowpal_{}".format(cb_type)] = Vowpal(
        num_actions=num_actions,
        vowpal_binary_path=args.vowpal_path,
        cb_type=cb_type,
        model_path=flexp.get_file_path(model_filename),
        load_model_path=(
            os.path.join(args.load_model_dir, model_filename)
            if args.load_model_dir
            else None
        )
    )

if not args.skip_train:
    logging.info("Training...")
    t = time.time()
    for chunk in train:
        chunk[["vowpal_train_input"]].to_csv(
            flexp.get_file_path("vowpal_input.txt"),
            index=False, header=None, sep="\t", mode="a"
        )
        logging.info(
            "timestamp {}, chunk took {:.3f} s to load"
            .format(chunk.timestamp.iloc[-1], time.time() - t)
        )

        _chunk["chunk"] = chunk
        t = time.time()
        with multiprocessing.Pool(len(models)) as pool:
            pool.map(update_model, models.keys())
        logging.info("updates took {:.3f} s".format(time.time() - t))
        t = time.time()

for cb_type in args.cb_types:
    models["vowpal_{}".format(cb_type)].stop()


def reshape(x):
    """Reshape a chunk column so that values for a SERP are in one row"""
    return x.values.reshape(-1, num_test_positions)


def cumany(x, axis=0):
    """Cumulative any (modeled after np.cumprod)"""
    return x.astype(bool).cumsum(axis=axis) > 0


def initial():
    return np.zeros(num_test_positions)


ndcg_numerator = defaultdict(initial)
ctr_numerator = defaultdict(initial)
vctr_numerator = defaultdict(initial)
c_sum = defaultdict(initial)
predictions = defaultdict(list)
propensities = defaultdict(list)
num_records = 0


t = time.time()
for chunk in test:
    if args.save_vowpal_input:
        chunk["vowpal_test_input"].to_csv(
            flexp.get_file_path("vowpal_test_input.txt"),
            index=False, header=None, sep="\t", mode="a"
        )

    _chunk["chunk"] = chunk
    logging.info(
        "timestamp {}, chunk took {:.3f} s to load"
        .format(chunk.timestamp.iloc[-1], time.time() - t)
    )

    t = time.time()
    preds = [models[k].get_action_probs_batch(chunk) for k in models.keys()]
    logging.info("getting predictions took {:.3f} s".format(time.time() - t))

    t = time.time()
    for key, new_predictions in zip(models.keys(), preds):
        new_propensities = new_predictions[
            range(len(chunk)), chunk.action
        ]
        chunk["{}_propensity".format(key)] = new_propensities

        if args.save_propensities:
            path = flexp.get_file_path(key + ".propensities.txt")
            with open(path, "a") as fout:
                for p in new_propensities:
                    print(p, file=fout)
    logging.info("appending propensities took {:.3f}".format(time.time() - t))

    t = time.time()
    for key in models.keys():
        # we assert that the number of loaded test positions is the same for
        # each SERP and reshape the propensities so that each SERP is in one
        # row
        new_propensity = reshape(chunk["{}_propensity".format(key)]).cumprod(1)

        # clip propensity after every cumulative product step to comply with
        # the awk version
        _old_propensity = reshape(chunk["propensity"])
        old_propensity = np.empty(_old_propensity.shape)
        min_prop = 1e-5 * np.ones(len(chunk) // num_test_positions)
        old_propensity[:, 0] = np.maximum(_old_propensity[:, 0], min_prop)
        for i in range(1, num_test_positions):
            old_propensity[:, i] = np.maximum(
                old_propensity[:, i - 1] * _old_propensity[:, i],
                min_prop
            )

        cs = new_propensity / old_propensity

        # see Section 4 of the paper for metric definitions
        ndcg_numerator[key] += (
            cs *
            (
                reshape(chunk["reward"] == 2) * np.log(2) /
                np.log(2 + np.arange(num_test_positions))
            ).cumsum(axis=1)
        ).sum(axis=0)

        ctr_numerator[key] += (
            cs * cumany(reshape(chunk["reward"] > 0), axis=1)
        ).sum(axis=0)

        vctr_numerator[key] += (
            cs *
            cumany(
                reshape(
                    (chunk["reward"] > 0) & (chunk["action"] != 0)
                ),
                axis=1
            )
        ).sum(axis=0)

        c_sum[key] += cs.sum(axis=0)
    num_records += len(chunk) // num_test_positions

    logging.info("evaluating metrics took {}".format(time.time() - t))


# save metric values
for i in range(num_test_positions):
    path = flexp.get_file_path("results_K={}.csv".format(i + 1))
    with open(path, "w") as fout:
        print("Policy", "CTR", "NDCG", "VCTR", "C", sep=";", file=fout)
        for key in models.keys():
            ndcg = ndcg_numerator[key][i] / c_sum[key][i]
            ctr = ctr_numerator[key][i] / c_sum[key][i]
            vctr = vctr_numerator[key][i] / c_sum[key][i]
            c = c_sum[key][i] / num_records
            print(key, ctr, ndcg, vctr, c, sep=";", file=fout)
            logging.info(
                "{} ndcg: {}, ctr: {}, vctr: {}, c: {}"
                .format(key, ndcg, ctr, vctr, c)
            )
