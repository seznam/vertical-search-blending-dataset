import os
import shutil
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE, DEVNULL
import logging

import numpy as np


class Policy:
    """Base class for policies"""
    def get_action_probs_batch(self, df):
        raise NotImplementedError

    def update(self, dataset):
        pass


class UniformPolicy(Policy):
    def __init__(self, num_actions):
        """
        :param num_actions: int  maximum number of available actions
        """
        self.num_actions = num_actions

    def get_action_probs(self, available_actions):
        """Get action probabilities given available actions

        :param available_actions: [int]  indices of available actions
        :return: np.array  Array of probabilities of chosing actions. Available
            actions with uniform probability, other actions with zero
            probability
        """
        probs = np.zeros(self.num_actions)
        probs[available_actions] = 1 / len(available_actions)
        return probs

    def get_action_probs_batch(self, df):
        """Get action probabilities for a whole pd.DataFrame
        """
        return np.array([
            self.get_action_probs(aa)
            for aa in df.available_actions
        ])


class Vowpal(Policy):
    def __init__(self, num_actions, vowpal_binary_path, model_path="model.vw",
                 load_model_path=None, cb_type="ips",
                 min_reward=0, max_reward=1, learn_from_scratch=True):
        """Vowpal contextual bandit model
        
        :param num_actions: int  maximum number of available actions
        :param vowpal_binary_path: str  path to the vowpal binary
        :param model_path: str  where the trained model should be saved
        :param load_model_path: str  start with this model
        :param cb_type: str  either dm (direct method), ips (inverse propensity
            score), or dr (doubly robust)
        :param min_reward: float  minimum possible reward
        :param max_reward: float  maximum possible reward
        :learn_from_scratch: bool  if not loading existing model and model_path
            exist, delete it
        """
        self.num_actions = num_actions
        self.vowpal_binary_path = vowpal_binary_path
        self.model_path = model_path
        self.cb_type = cb_type
        self.min_reward = min_reward
        self.max_reward = max_reward

        # for reading vowpal output asynchronically
        self.reading_threadpool = ThreadPool(1)

        if load_model_path is not None:
            shutil.copyfile(load_model_path, model_path)
        elif learn_from_scratch and os.path.isfile(model_path):
            os.remove(model_path)

        self.start("train")

    def start(self, mode):
        """Start vowpal process

        :param mode: str  either "train" (throw away vowpal output), or
            "predict"
        """
        assert mode in ["train", "predict"]
        self.mode = mode

        args = [
            self.vowpal_binary_path,
            "-f", self.model_path, "-p", "/dev/stdout", "--quiet"
        ]

        if not os.path.isfile(self.model_path):
            # these options get saved into the model
            args += ["--cb", str(self.num_actions), "--cb_type", self.cb_type]
        else:
            args += ["-i", self.model_path]

        logging.info("Vowpal command: {}".format(" ".join(args)))

        self.process = Popen(
            args,
            stdout=DEVNULL if mode == "train" else PIPE,
            stdin=PIPE,
            bufsize=1,
            universal_newlines=True
        )

    def stop(self):
        """Stop vowpal process"""
        if self.mode is None:
            return
        self.process.stdin.flush()
        self.process.stdin.close()
        if self.mode == "predict":
            self.process.stdout.close()
        self.process.wait()
        self.mode = None

    def __del__(self):
        if self.mode is not None:
            self.stop()
        self.reading_threadpool.close()
        self.reading_threadpool.join()

    def read_predictions(self):
        """Read predictions for one position from the vowpal process
        
        :return: np.array  Array of probabilities of chosing actions. Chosen
            action with probability 1, other actions with zero probability
            (vowpal contextual bandit models are deterministic).
        """
        predictions = self.process.stdout.readline().strip().split()
        while predictions[0] == "warning:":
            logging.warning("Vowpal says: {}".format(" ".join(predictions)))
            predictions = self.process.stdout.readline().strip().split()
        action = int(predictions[0]) - 1
        probs = np.zeros(self.num_actions)
        probs[action] = 1
        return probs

    def read_predictions_batch(self, n):
        return [self.read_predictions() for _ in range(n)]

    def get_action_probs_batch(self, df):
        """Get action probabilities for a whole pd.DataFrame
        """
        if self.mode != "predict":
            self.stop()
            self.start("predict")

        if "vowpal_test_input" not in df:
            return super().get_action_probs_batch(df)

        async_result = self.reading_threadpool.apply_async(
            self.read_predictions_batch,
            args=(len(df),)
        )
        for line in df["vowpal_test_input"]:
            print(line, file=self.process.stdin)
        probs = async_result.get()
        return np.array(probs)

    def update(self, df):
        """Train model on df

        :param df: pd.DataFrame  should contain "vowpal_train_input" column
        """
        if self.mode != "train":
            self.stop()
            self.start("train")

        vowpal_input = df["vowpal_train_input"]

        for line in vowpal_input:
            print(line, file=self.process.stdin)
        self.process.stdin.flush()
