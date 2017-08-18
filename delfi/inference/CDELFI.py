import delfi.distribution as dd
import numpy as np
import theano.tensor as tt

from delfi.inference.BaseInference import InferenceBase
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero


class CDELFI(InferenceBase):
    def __init__(self, generator, obs, n_components=1, reg_lambda=100.,
                 seed=None, **kwargs):
        """Conditional density estimation likelihood-free inference (CDE-LFI)

        Implementation of algorithms 1 and 2 of Papamakarios and Murray, 2016.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        n_components : int
            Number of components in final round (PM's algorithm 2)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        # Algorithm 1 of PM requires a single component
        kwargs.update({'n_components': 1})

        super().__init__(generator, seed=seed, **kwargs)

        self.n_components = n_components
        self.obs = obs
        self.reg_lambda = reg_lambda

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        if self.svi:
            kl = svi_kl_zero(self.network.mps, self.network.sps,
                             self.reg_lambda)
            loss = loss + 1 / N * kl

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.kl'] = kl

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=1000, minibatch=50,
            monitor=None, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int
            Number of data points drawn per round
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training
        minibatch: int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets
        """
        logs = []
        trn_datasets = []

        for r in range(1, n_rounds + 1):  # start at 1
            trn_data = self.gen(n_train)  # z-transformed params and stats

            # algorithm 2 of Papamakarios and Murray
            if r == n_rounds and self.n_components > 1:
                # get parameters of current network
                old_params = self.network.params_dict.copy()

                # create new network
                network_spec = self.network.spec_dict.copy()
                network_spec.update({'n_components': self.n_components})
                self.network = NeuralNet(**network_spec)
                new_params = self.network.params_dict

                # set weights of new network
                # weights of additional components are duplicates
                for p in [
                        s for s in new_params if 'means' in s or 'precisions' in s]:
                    new_params[p] = old_params[p[:-1] + '0']
                self.network.params_dict = new_params

            t = Trainer(self.network, self.loss(N=n_train), trn_data,
                        monitor=self.monitor_dict_from_names(monitor),
                        seed=self.gen_newseed(), **kwargs)

            logs.append(t.train(epochs=epochs, minibatch=minibatch))
            trn_datasets.append(trn_data)

            # posterior becomes new proposal prior
            posterior = self.predict(self.obs)
            self.generator.proposal = posterior.project_to_gaussian()

        return logs, trn_datasets

    def predict(self, x):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        if self.generator.proposal is None:
            # no correction necessary
            return super(CDELFI, self).predict(x)  # via super
        else:
            # mog is posterior given proposal prior
            mog = super(CDELFI, self).predict(x)  # via super

            # compute posterior given prior by analytical division step
            if 'Uniform' in str(type(self.generator.prior)):
                posterior = mog / self.generator.proposal
            elif 'Gaussian' in str(type(self.generator.prior)):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented

            return posterior