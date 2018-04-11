import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError

        # initialise with defaut value
        bn = self.n_constant
        bic = math.inf

        try:
            # n between self.min_n_components and self.max_n_components
            for n in range(self.min_n_components, self.max_n_components+1):   
                # LogL : log likelihood of fitted model
                base_model = self.base_model(n)
                logL = base_model.score(self.X, self.lengths)
                # p : number of parameters
                p = n**2 + 2*n*base_model.n_features - 1
                # bic = 2 * logL + p * logN
                score = 2*logL + p*(math.log(n))
                if score < bic:
                    bn = n
                    bic = score
        except:
            if self.verbose:
                print("failure")
        return self.base_model(bn)


class SelectorDIC(ModelSelector):
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError        

        # initialize 
        bn = self.n_constant
        dic = -math.inf
        logsL = list()

        try:
            for n in range(self.min_n_components, self.max_n_components + 1):
                # List of LogL
                base_model = self.base_model(n)
                logsL.append(base_model.score(self.X, self.lengths))

            # Total words
            M = self.max_n_components - self.min_n_components + 1
            logL_sum = sum(logsL)

            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            for n in range(self.min_n_components, self.max_n_components + 1):
                logL = logsL[n - self.min_n_components]
                logL_bar = logL_sum - logL
                score = logL - (logL_bar / (M-1))
                if score > dic:
                    dic = score
                    bn = n
        except: 
            if self.verbose:
                print("failure")
        return self.base_model(bn)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #raise NotImplementedError

        best_score = -math.inf
        bn = self.n_constant
        splits = KFold()

        try:
            for n in range(self.min_n_components, self.max_n_components+1):
                folds = None
                for train_split, test_split in splits.split(self.sequences):  
                    train, train_len = combine_sequences(train_split, self.sequences)
                    test, test_len = combine_sequences(test_split, self.sequences)
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train, train_len)
                    folds.append(model.score(test, test_len))
                mean = np.mean(folds)
                if mean > best_score:
                    best_score = mean
                    bn = n
        except :
            if self.verbose:
                print("failure")

        return self.base_model(bn)

         
