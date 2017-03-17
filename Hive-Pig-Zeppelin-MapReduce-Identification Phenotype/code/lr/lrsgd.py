# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu
        self.n = n_feature

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        # last_error = float('inf')
        # error = sum((self.predict(X)-y)**2)/n
        # threshold = 10^(-6)
        # max_iter = 10^10
        # iteration = 0

        # while abs(last_error-error)>threshold and iteration<max_iter:
        #     for f,v in X:
        #         self.weight[f] = self.weight[f] - self.eta * ((self.predict_prob(X)-y)*X + 2*self.mu*self.weight)
        #     last_error = error
        #     error = sum((self.predict(X)-y)**2)/n
        #     iteration += 1    
        for i in range(len(X)):
            self.weight[X[i][0]] = self.weight[X[i][0]] - self.eta * ((self.predict_prob(X)-y)*X[i][1] + 2*self.mu*self.weight[X[i][0]])


    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        result = 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
        return result
