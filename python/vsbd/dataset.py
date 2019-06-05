import multiprocessing
import logging
from itertools import chain

import pandas as pd


def list_chain(it):
    return list(chain.from_iterable(it))


class DatasetReader:
    """Dataset reader

    Instance's read() method reads (parts of) the dataset into pd.DataFrame. As
    the result would be usually too big, it can be also loaded by chunks.
    """
    @staticmethod
    def pos_i_cols(i):
        """Return list of pairs (column name, dtype) for features at a given
        position

        :param i: int  position
        :return: [(str, dtype)]  columns description
        """
        return [
            (x.format(i), d)
            for x, d in [
                ("reward_{}", float),
                ("propensity_{}", float),
                ("action_{}", str),  # int
                ("domain_{}", str)
            ]
        ]

    # features common to the whole SERP
    serp_columns = [
        ("id", str),
        ("query", str),
        ("num_tokens", int),
        ("num_skips", int),
        ("timestamp", str),
        ("alternative_actions", str),
        ("hardware", str),
    ]

    def __init__(self, path, nrows=None, chunksize=None, use_positions=None,
                 skip_transform=False, make_vowpal_train_input=False,
                 make_vowpal_test_input=False, dtype=None, clip_bound=1e-5,
                 njobs=5):
        """
        :param path: str or [str]  path(s) to dataset files
        :param nrows: int  only that many first lines from each file
        :param chunksize: int  don't read everything in memory at once –
            iterate over chunks instead
        :param use_positions: [int]  only read these SERP positions (0 indexed)
        :param skip_transform: bool  don't transform – return the pd.DataFrame
            as read by pd.read_csv
        :param make_vowpal_train_input: bool  add column "vowpal_train_input"
        :param make_vowpal_test_input: bool  add column "vowpal_test_input"
        :param dtype: pd.read_csv dtype specification
        :param clip_bound: float  clip propensities lower than this
        :param njobs: int  number of processes preparing chunks in parallel,
            None for no parallelism
        """

        if isinstance(path, str):
            path = [path]
        self.paths = path
        self.use_positions = use_positions
        self.full_pos_columns = list_chain(
            self.pos_i_cols(i) for i in range(14)
        )
        self.nrows = nrows
        self.chunksize = chunksize
        self.skip_transform = skip_transform
        self.dtype = dtype
        self.make_vowpal_train_input = make_vowpal_train_input
        self.make_vowpal_test_input = make_vowpal_test_input
        self.njobs = njobs
        self.clip_bound = clip_bound

    def _add_vowpal_context(self, df):
        date = pd.to_datetime(df.timestamp, format="%Y-%m-%d-%H-%M-%S-%Z")
        df["vowpal_context"] = [
            "q={} nt:{} ns={} hw={} wd={} h={} {}p:{}".format(*x)
            for x in zip(
                df["query"].fillna(""),
                df.num_tokens, df.num_skips,
                df.hardware,
                date.dt.weekday, date.dt.hour,
                df.domain.fillna("").map(
                    lambda x: "d={} ".format(x) if x else x
                ),
                df.position
            )
        ]
        return df

    def _add_vowpal_train_input(self, df):
        min_reward = 0
        max_reward = 1
        clicks = df.reward > 0
        norm_rewards = (
            (clicks - min_reward) /
            (max_reward - min_reward)
        )
        d = 0.00001
        # cost is `d` if clicked, (1 - `d`) otherwise
        costs = 1 - (norm_rewards * (1 - 2 * d) + d)

        labels = []
        for cost, action, available_actions, propensity in zip(
                costs, df.action, df.available_actions, df.propensity):
            smaller = " ".join(
                str(a + 1) for a in available_actions if a < action
            )
            if smaller:
                smaller += " "
            greater = " ".join(
                str(a + 1) for a in available_actions if a > action
            )
            if greater:
                greater = " " + greater
            labels.append(
                "{}{}:{:.5f}:{:.5f}{}".format(
                    smaller, action + 1, cost, propensity, greater
                )
            )

        df["vowpal_label"] = labels

        self._add_vowpal_context(df)
        df["vowpal_train_input"] = (
            df["vowpal_label"] + " | " + df["vowpal_context"]
        )
        return df

    def _add_vowpal_test_input(self, df):
        df["vowpal_available_actions"] = [
            " ".join(str(a + 1) for a in aa)
            for aa in df.available_actions
        ]

        self._add_vowpal_context(df)
        df["vowpal_test_input"] = (
            df["vowpal_available_actions"] + " | " + df["vowpal_context"]
        )
        return df

    def _transform(self, df):
        df_long = pd.wide_to_long(
            df,
            ["reward", "propensity", "action", "domain"],
            i="id", j="position", suffix="_\\d+"
        ).reset_index()
        df_long["position"] = df_long["position"].str.replace("_", "").map(int)
        df_long = df_long[~df_long["action"].isna()].sort_values(
            ["timestamp", "id", "position"]
        )
        df_long.reset_index(inplace=True, drop=True)
        df_long["action"] = pd.to_numeric(df_long["action"]).astype(int)

        def available_actions(alternative_actions, actions_taken):
            current_actions = set(map(int, [0] + alternative_actions.split()))
            last_vertical_position = -4
            for i, at in enumerate(actions_taken):
                if not isinstance(at, str):
                    break
                if i - last_vertical_position > 3:
                    res = sorted(current_actions)
                    if int(at) != 0:
                        current_actions.remove(int(at))
                        last_vertical_position = i
                else:
                    res = [0]
                yield res

        df_long["available_actions"] = list_chain(
            available_actions(aa, at) for aa, at in zip(
                df["alternative_actions"],
                df.filter(regex="action_.*").values
            )
        )
        df = df_long

        if self.clip_bound:
            num_small_prop = (df["propensity"] < self.clip_bound).sum()
            df["propensity"].clip(self.clip_bound, 1, inplace=True)
            if num_small_prop > 0:
                logging.warning(
                    "Clipped {} too small propensities"
                    .format(num_small_prop)
                )

        columns = [
            "id", "action", "available_actions", "propensity", "reward",
            "timestamp"
        ]
        if self.make_vowpal_train_input:
            self._add_vowpal_train_input(df)
            columns.append("vowpal_train_input")
        if self.make_vowpal_test_input:
            self._add_vowpal_test_input(df)
            columns.append("vowpal_test_input")

        return df_long[columns]

    @staticmethod
    def _limit_buffer_size(df, semaphore):
        for chunk in df:
            semaphore.acquire()
            yield chunk

    def read(self):
        pos_columns = self.full_pos_columns
        if self.use_positions:
            pos_columns = list_chain(
                self.pos_i_cols(i) for i in self.use_positions
            )

        df = chain.from_iterable(
            pd.read_csv(
                path, sep="\t",
                names=[
                    x for x, _ in self.serp_columns + self.full_pos_columns
                ],
                low_memory=False,
                nrows=self.nrows,
                chunksize=self.chunksize,
                usecols=[x for x, _ in self.serp_columns + pos_columns],
                dtype=(
                    dict(self.serp_columns + pos_columns)
                    if self.dtype is None else self.dtype
                ),
                na_values="",
                keep_default_na=False
            )
            for path in self.paths
        )

        if self.skip_transform:
            return df

        if self.njobs is not None:
            semaphore = multiprocessing.Semaphore(self.njobs + 3)
            lim_df = self._limit_buffer_size(df, semaphore)
            with multiprocessing.Pool(self.njobs) as pool:
                for chunk in pool.imap(self._transform, lim_df):
                    yield chunk
                    semaphore.release()
        else:
            for chunk in map(self._transform, df):
                yield chunk
