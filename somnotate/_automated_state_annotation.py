#!/usr/bin/env python
"""
Automated state annotation using a combination of linear
discriminant analysis (LDA) and hidden Markov model (HMM).
"""

import pickle
import numpy as np

from pomegranate import (
    MultivariateGaussianDistribution,
    HiddenMarkovModel,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class StateAnnotator(object):

    def __init__(self):
        pass

    def fit(self, signal_arrays, state_vectors):

        """Given a set of signals and corresponding state annotations,
        train the annotator to predict the states based on the signals.

        Arguments:
        ----------
        signal_arrays -- list of (total samples, data dimensions) ndarrays of floats
            The collection multidimensional time series signals.
            The data dimensions need to match across signal arrays but the total
            samples in each signal array can vary between arrays.

        state_vectors -- list of (total samples, ) vectors of integers
            The corresponding state for each sample in each signal array,
            represented by an integer.

            Positive integers (1, 2, 3, ... ) are used to denote the
            state of valid samples.  The corresponding samples are
            used to train the linear discriminant.

            Negative integers (-1, -2, -3, ...) are used to denote the
            state of invalid samples. These are samples with a defined
            state, which nevertheless should not be used in training
            of the linear discriminant, as they are, for example,
            outliers or artefacts. These states are used, however, to
            determine the transition probabilities between states.

            Samples with undefined states are denoted by zeros. These
            are used in neither the training of the linear
            discriminant nor the HMM (NaNs cannot be represented by
            ndarrays of dtype int).
        """

        self._check_inputs(signal_arrays, state_vectors)

        self.transformer = self.fit_transform(signal_arrays, state_vectors,
                                      solver='eigen', shrinkage='auto')

        # We want to bunch together artefact states with their
        # corresponding "clean" states.
        state_vectors = [np.abs(vec) for vec in state_vectors]

        self.hmm = self.fit_hmm(signal_arrays, state_vectors,
                                MultivariateGaussianDistribution)


    def transform(self, arr):
        return self.transformer.transform(arr)


    def predict(self, signal_array):
        """
        Predict the states for each sample in X.

        Arguments:
        ----------
        signal_array -- (total samples, data dimensions) ndarray of floats
            A multidimensional time series signal.

        Returns:
        --------
        state_vector -- iterable of state labels
            The predicted states, one for each sample in the signal array.

        """

        self._check_signal_array(signal_array)

        transformed_array = self.transform(signal_array)

        _, viterbi_path = self.hmm.viterbi([sample for sample in transformed_array])

        # viterbi path calculation may fail -- when it does, it does so silently!
        assert viterbi_path != None, "Viterbi path calculation failed."

        # trim start and end states
        _, state = viterbi_path[0]
        if state.name == "None-start":
            viterbi_path = viterbi_path[1:]

        _, state = viterbi_path[-1]
        if state.name == "None-end":
            viterbi_path = viterbi_path[:-1]

        predicted_states = [state.name for _  , state in viterbi_path]

        # pomegranate only allows strings as state labels, but we use intergers
        predicted_states = [int(state) for state in predicted_states]

        return predicted_states


    def predict_proba(self, signal_array):
        """
        Predict the probability of the predicted state for each sample in
        the signal array given all possible paths through the state
        space.

        Arguments:
        ----------
        signal_array -- (total samples, data dimensions) ndarray
            The multidimensional time series signal.

        Returns:
        --------
        p    -- dict state : (total samples, ) vector
            A mapping of each state to a vector of probabilities.

        """

        self._check_signal_array(signal_array)
        transformed_array = self.transform(signal_array)

        probability_array = self.hmm.predict_proba([sample for sample in transformed_array])

        # convert array to dictionary
        probability_dict = dict()
        for ii, state in enumerate(self.hmm.states):
            if not (state.distribution is None): # i.e. ignore start and end states, which have 100% probability at start and end and zero elsewhere
                state_label = int(state.name)
                probability_dict[state_label] = probability_array[:, ii]

        # for each sample, extract the probability corresponding to the predicted state
        predicted_state_vector = self.predict(signal_array)
        probability_vector = np.array([probability_dict[state][ii] for ii, state in enumerate(predicted_state_vector)])

        return probability_vector


    def score(self, signal_array, state_vector):
        self._check_inputs([signal_array], [state_vector])
        predicted_state_vector = self.predict(signal_array)
        return np.mean(predicted_state_vector == state_vector)


    def _check_inputs(self, signal_arrays, state_vectors):
        # check that everything has the correct type and dimension
        if not isinstance(signal_arrays, list):
            raise TypeError("`signal_arrays` has to be a list! Current type: {}".format(type(signal_arrays)))

        if not isinstance(state_vectors, list):
            raise TypeError("`state_vectors` has to be a list! Current type: {}".format(type(state_vectors)))

        for arr in signal_arrays:
            self._check_signal_array(arr)

        for vec in state_vectors:
            self._check_state_vector(vec)

        # check that dimensions match where they need to match
        for ii, (arr, vec) in enumerate(zip(signal_arrays, state_vectors)):
            assert len(arr) == len(vec), "The lengths of the signal array ({}) and the corresponding state vector ({}) do not match!".format(len(arr), len(vec))

        signal_dimensions = []
        for arr in signal_arrays:
            if arr.ndim == 1:
                signal_dimensions.append(1)
            else:
                signal_dimensions.append(arr.shape[1])

        # signal_dimensions = [arr.shape[1] if arr.ndim == 1 else 1 for arr in signal_arrays]
        assert len(set(signal_dimensions)) == 1, "The second dimension of all signal arrays need to match! Current dimensions: {}".format(signal_dimensions)


    def _check_signal_array(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError("Signal arrays have to be an instance of numpy.ndarray. Current type: {}".format(type(arr)))

        if not (arr.dtype in (int, float)):
            raise TypeError("Elements of signal arrays have to be of type int or float. Current type: {}".format(arr.dtype))

        assert arr.ndim >= 1, "Signal arrays have to have at least 1 dimension. Current dimensionality: {}".format(arr.ndim)


    def _check_state_vector(self, vec):
        if not isinstance(vec, np.ndarray):
            raise TypeError("State vectors have to be an instance of numpy.ndarray. Current type: {}".format(type(vec)))

        if not (vec.dtype in (int,)):
            raise TypeError("Elements of state vectors have to be of type int. Current type: {}".format(vec.dtype))

        assert vec.ndim == 1, "State vectors have to have exactly 1 dimension. Current dimensionality: {}".format(vec.ndim)


    def save(self, file_path):
        objects = dict(transformer=self.transformer, hmm=self.hmm.to_json())
        with open(file_path, 'wb') as f:
            pickle.dump(objects, f)


    def load(self, file_path):
        with open(file_path, 'rb') as f:
            objects = pickle.load(f)
        try:
            self.transformer = objects['transformer']
        except KeyError: # for backwards compatibility
            self.transformer = objects['lda']
        self.hmm = HiddenMarkovModel.from_json(objects['hmm'])


    def fit_transform(self, signals, states, **kwargs):
        # Ultra-thin wrapper around sklearn.discriminant_analysis.LinearDiscriminantAnalysis.fit()
        # that primarily serves to decouple the interface from the sklearn implementation.

        # combine data sets
        combined_signals = np.concatenate(signals, axis=0)
        combined_states = np.concatenate(states, axis=0)

        # remove undefined / artefact states
        is_valid = combined_states > 0
        combined_signals = combined_signals[is_valid]
        combined_states = combined_states[is_valid]

        return LinearDiscriminantAnalysis(**kwargs).fit(combined_signals, combined_states)


    def fit_hmm(self, signal_arrays, state_vectors, distribution, **kwargs):

        signals = [self.transform(arr) for arr in signal_arrays]

        # pomegranate expects string labels for states
        labels = [[str(state) for state in vec] for vec in state_vectors]

        # cosntruct matching state names
        state_names = [str(state) for state in np.unique(np.concatenate(state_vectors))]

        hmm = HiddenMarkovModel.from_samples(
            distribution = distribution,
            n_components = len(state_names),
            X            = signals,
            labels       = labels,
            algorithm    = 'labeled',
            state_names  = state_names,
            **kwargs)

        # TODO:
        # - remove state corresponding to "undefined"
        # - remove edges below state transition threshold
        # - adjust start probabilities based on data

        return hmm
