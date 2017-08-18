import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc


class SummaryStatsBase(metaclass=ABCMetaDoc):
    def __init__(self, seed=None):
        """Abstract base class for summary stats

        Summary Stats must at least implement abstract methods and properties of
        this class.

        Parameters
        ----------
        seed : int or None
            If provided, random number generator will be seeded

        Attributes
        ----------
        n_summary : int
            Number of resulting summary features
        """
        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed
        self.n_summary = None

    @abc.abstractmethod
    def calc(self, repetition_list):
        """Method computing summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.arrray, 2d with n_reps x n_summary
        """
        pass

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)