import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc


class BaseGenerator(metaclass=ABCMetaDoc):
    def __init__(self, model, prior, summary):
        """Generator

        Parameters
        ----------
        model : Simulator instance
            Forward model
        prior : Distribution or Mixture instance
            Prior over parameters
        summary : SummaryStats instance
            Summary statistics

        Attributes
        ----------
        proposal : None or Distribution or Mixture instance
            Proposal prior over parameters. If specified, will generate
            samples given parameters drawn from proposal distribution rather
            than samples drawn from prior when `gen` is called.
        """
        self.model = model
        self.prior = prior
        self.summary = summary
        self.proposal = None

    def gen(self, n_samples, n_reps=1, skip_feedback=False, verbose=True):
        """Draw parameters and run forward model

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_reps: int, default=1
            Number of repetitions per parameter sample, useful for stochastic simulations
        skip_feedback: bool
            If True, feedback checks on params, data and sum stats are skipped
        verbose : bool or str
            If False, will not display progress bars. If a string is passed,
            the string will be set as a description for the progress bar.

        Returns
        -------
        params : n_samples x n_reps x n_params
            Parameters
        stats : n_samples x n_reps x n_summary
            Summary statistics of data
        """
        assert n_reps == 1, 'n_reps > 1 is not yet supported'

        # collect valid parameter vectors from the prior
        params = []  # list of parameter vectors
        i = 0
        while i < n_samples:
            # sample parameter
            if self.proposal is None:
                proposed_param = self.prior.gen(n_samples=1)  # dim params,
            else:
                proposed_param = self.proposal.gen(n_samples=1)

            # check if parameter vector is valid
            response = self._feedback_proposed_param(proposed_param)
            if response == 'accept' or skip_feedback:
                # add valid param vector to list
                params.append(proposed_param.reshape(-1))
                i += 1
            elif response == 'resample':
                continue  # continue without increment on i
            else:
                raise ValueError('response not supported')

        # run forward model for all params, each n_reps times
        result = self.model.gen(params, n_reps=n_reps, verbose=verbose)

        # for every datum in data, check validity
        params_data_valid = []  # list of params with valid data
        data_valid = []  # list of lists containing n_reps dicts with data
        for param, datum in zip(params, result):
            # check validity
            response = self._feedback_forward_model(datum)
            if response == 'accept' or skip_feedback:
                data_valid.append(datum)
                # if data is accepted, accept the param as well
                params_data_valid.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        # for every data in data, calculate summary stats
        final_params = []
        final_stats = []  # list of summary stats
        for param, datum in zip(params_data_valid, data_valid):
            # calculate summary statistics
            sum_stats = self.summary.calc(datum)  # n_reps x dim stats

            # check validity
            response = self._feedback_summary_stats(sum_stats)
            if response == 'accept' or skip_feedback:
                final_stats.append(sum_stats)
                # if sum stats is accepted, accept the param as well
                final_params.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        # TODO: for n_reps > 1 duplicate params; reshape stats array
        params = np.array(final_params)  # n_samples x n_reps x dim theta
        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze(axis=1)

        return params, stats

    @abc.abstractmethod
    def _feedback_proposed_param(self, param):
        """Feedback step after parameter has been proposed

        Parameters
        ----------
        param : np.array
            Parameter

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'resample']
        """
        # TODO: check if parameter is inside of support of prior when
        # proposal distribution was used for sampling
        return 'accept'

    @abc.abstractmethod
    def _feedback_forward_model(self, data):
        """Feedback step after forward model ran

        Parameters
        ----------
        data : np.array
            Data

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'discard']
        """
        return 'accept'

    @abc.abstractmethod
    def _feedback_summary_stats(self, sum_stats):
        """Feedback step after summary stats were computed

        Parameters
        ----------
        sum_stats : np.array
            Summary stats

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'discard']
        """
        return 'accept'
