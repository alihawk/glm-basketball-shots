import numpy as np
from scipy.optimize import fmin_l_bfgs_b  # Optimizer used for minimizing the negative log-likelihood
from math import exp, log  # For exponential and logarithm calculations


def _softmax(logits):
    """
    Compute the softmax function row-wise for a 2D array of logits.

    Parameters:
      logits (np.array): A 2D numpy array of shape (N, K) containing the unnormalized log probabilities.

    Returns:
      np.array: A 2D numpy array of shape (N, K) where each row sums to 1, representing a probability distribution.
    """
    # Compute the exponential of each element in the logits array.
    exps = np.exp(logits)
    # Compute the sum of the exponentials for each row, keeping dimensions for broadcasting.
    sums = np.sum(exps, axis=1, keepdims=True)
    # Divide each exponential value by the sum of its corresponding row to obtain probabilities.
    return exps / sums


class MultinomialLogReg:
    """
    Multinomial Logistic Regression using maximum likelihood estimation.

    This model is used for multi-class classification where the response variable has K classes.
    One of the classes (typically the last) is set as the baseline (its coefficients are fixed to zero)
    to address the identifiability issue.

    Example usage:
      model = MultinomialLogReg()
      model.build(X, y)         # Fit the model on data X with labels y
      probs = model.predict(X_new)  # Predict class probabilities for new data X_new
    """

    def __init__(self):
        # Number of classes (K) will be determined during fitting
        self.num_classes_ = None
        # Number of features (d) will be determined during fitting
        self.num_features_ = None
        # Model parameters (coefficients) will be stored here after fitting.
        # They have shape (d * (K-1)), since we only estimate parameters for (K-1) classes.
        self.params_ = None

    def build(self, X, y):
        """
        Fit the multinomial logistic regression model to data X and target y using Maximum Likelihood Estimation.

        Parameters:
          X (array-like): The feature matrix of shape (N, d).
          y (array-like): The target vector with integer class labels of shape (N,).

        Returns:
          self: The fitted model object with learned parameters.
        """
        # Convert X and y into numpy arrays with appropriate data types.
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Get number of samples (N) and number of features (d)
        N, d = X.shape
        # Get the unique classes in y and count them to determine K
        classes = np.unique(y)
        self.num_classes_ = len(classes)
        self.num_features_ = d

        # Use a baseline approach: fix the coefficients for the last class to zero.
        # This means we only need to optimize parameters for the first (K-1) classes.
        K = self.num_classes_

        # Initialize all parameters (for K-1 classes) to zeros.
        # Total number of parameters is (K-1) * d.
        init_params = np.zeros((K - 1) * d)

        # Define the negative log-likelihood function to be minimized.
        def neg_loglik(params):
            # Reshape the parameter vector into a matrix of shape (d, K-1)
            # Each column of W corresponds to the coefficients for one non-baseline class.
            W = params.reshape(d, K - 1)
            # Create an array to hold logits for all samples and all K classes.
            # Initialize with zeros; for the baseline class (last column) logits remain 0.
            logits = np.zeros((N, K))
            # For the first (K-1) classes, compute logits as the dot product of X and the corresponding column of W.
            logits[:, :-1] = X.dot(W)
            # The logits for the baseline class (column K-1) remain zero.

            # Convert the logits into probabilities using the softmax function.
            probs = _softmax(logits)

            # Compute the negative log-likelihood.
            # For each sample, take the logarithm of the predicted probability for the true class,
            # add a small epsilon (1e-15) for numerical stability, and sum the negative logs over all samples.
            nll = 0.0
            for i in range(N):
                nll -= log(probs[i, y[i]] + 1e-15)
            return nll

        # Optimize the negative log-likelihood using fmin_l_bfgs_b.
        # We use numerical gradients by setting approx_grad=True.
        result = fmin_l_bfgs_b(
            func=neg_loglik,  # Function to minimize
            x0=init_params,  # Initial guess for the parameters
            approx_grad=True  # Use numerical approximation for the gradient
        )
        # Store the optimized parameters.
        self.params_ = result[0]
        return self

    def predict(self, X):
        """
        Predict class probabilities for new input data X.

        Parameters:
          X (array-like): New feature matrix of shape (N, d).

        Returns:
          np.array: An array of predicted probabilities of shape (N, K), where each row sums to 1.
        """
        # Convert X to a numpy array.
        X = np.array(X, dtype=float)
        # Determine number of samples (N) and number of features (d)
        N, d = X.shape
        # Retrieve the number of classes (K) learned during model fitting.
        K = self.num_classes_

        # Reshape the learned parameters into a matrix of shape (d, K-1)
        W = self.params_.reshape(d, K - 1)

        # Initialize an array to hold logits for all samples and classes.
        logits = np.zeros((N, K))
        # For the first (K-1) classes, compute logits as the dot product of X and W.
        logits[:, :-1] = X.dot(W)
        # The baseline class logits (column K-1) remain zero.

        # Convert the logits into probabilities using the softmax function.
        probs = _softmax(logits)
        return probs


class OrdinalLogReg:


    def __init__(self):
        # Number of classes (K) will be determined during model fitting.
        self.num_classes_ = None
        # Number of features (d) will be determined during model fitting.
        self.num_features_ = None
        # Parameters will contain the concatenated vector [beta, threshold_params].
        # Total length is d + (K-1).
        self.params_ = None

    def build(self, X, y):
        """
        Fit the ordinal logistic regression model to data X and target y.

        Parameters:
          X (array-like): Feature matrix of shape (N, d).
          y (array-like): Ordered target vector of shape (N,).

        Returns:
          self: The fitted model object with learned parameters.
        """
        # Convert inputs to numpy arrays with appropriate types.
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Get number of samples (N) and number of features (d)
        N, d = X.shape
        # Determine unique classes in y and count them to get K.
        classes = np.unique(y)
        K = len(classes)
        self.num_classes_ = K
        self.num_features_ = d

        # We need to estimate:
        #   - d coefficients (beta) for the linear predictor,
        #   - (K-1) threshold parameters to divide the latent variable's scale.
        # Total number of parameters is d + (K-1).
        init_params = np.zeros(d + (K - 1))

        # Define the logistic cumulative distribution function (CDF).
        def logistic_cdf(z):
            return 1.0 / (1.0 + np.exp(-z))

        # Define the negative log-likelihood function for ordinal regression.
        def neg_loglik(params):
            # Extract the beta coefficients from the first d elements.
            beta = params[:d]
            # Extract the raw threshold parameters from the remaining elements.
            threshold_raw = params[d:]

            # Reconstruct thresholds to enforce increasing order.
            # Set b[0] = threshold_raw[0], then b[i] = b[i-1] + exp(threshold_raw[i]) for i >= 1.
            b = []
            current = threshold_raw[0]
            b.append(current)
            for i in range(1, K - 1):
                # Exponentiating ensures the increment is positive.
                current = current + np.exp(threshold_raw[i])
                b.append(current)
            # Convert the threshold list to a numpy array.
            b = np.array(b)  # This array has shape (K-1,)

            # Compute the linear predictor (eta) for each sample.
            eta = X.dot(beta)  # Shape: (N,)

            # Initialize the negative log-likelihood.
            nll = 0.0
            # For each sample, compute the probability of the observed category.
            for i in range(N):
                xi = eta[i]
                yi = y[i]
                # For the first category:
                if yi == 0:
                    p = logistic_cdf(b[0] - xi)
                # For the last category:
                elif yi == (K - 1):
                    p = 1.0 - logistic_cdf(b[K - 2] - xi)
                # For intermediate categories:
                else:
                    p = logistic_cdf(b[yi] - xi) - logistic_cdf(b[yi - 1] - xi)
                # Accumulate the negative log of the probability, adding a small epsilon for stability.
                nll -= log(p + 1e-15)
            return nll

        # Optimize the negative log-likelihood using fmin_l_bfgs_b.
        result = fmin_l_bfgs_b(
            func=neg_loglik,  # Function to minimize
            x0=init_params,  # Initial parameter guess (all zeros)
            approx_grad=True  # Use numerical gradient estimation
        )
        # Store the optimized parameters.
        self.params_ = result[0]
        return self

    def predict(self, X):
        """
        Predict the probabilities for each ordinal category given new data X.

        Parameters:
          X (array-like): New feature matrix of shape (N, d).

        Returns:
          np.array: Predicted probability matrix of shape (N, K), where each row sums to 1.
        """
        # Convert X to a numpy array.
        X = np.array(X, dtype=float)
        # Get number of samples (N) and features (d)
        N, d = X.shape
        # Retrieve the number of classes (K)
        K = self.num_classes_

        # Extract beta coefficients from the first d elements of the learned parameters.
        beta = self.params_[:d]
        # Extract the raw threshold parameters.
        threshold_raw = self.params_[d:]

        # Reconstruct the thresholds in the same way as in the build method.
        b_vals = []
        current = threshold_raw[0]
        b_vals.append(current)
        for i in range(1, K - 1):
            current = current + np.exp(threshold_raw[i])
            b_vals.append(current)
        b_vals = np.array(b_vals)  # b_vals now has shape (K-1,)

        # Define the logistic CDF function.
        def logistic_cdf(z):
            return 1.0 / (1.0 + np.exp(-z))

        # Compute the linear predictor for each sample.
        eta = X.dot(beta)  # Shape: (N,)

        # Initialize an array to hold the predicted probabilities for all N samples and K classes.
        probs = np.zeros((N, K))
        # For each sample, compute the probability of each ordinal category.
        for i in range(N):
            xi = eta[i]
            # First category probability: P(Y=0) = F(b1 - xi)
            probs[i, 0] = logistic_cdf(b_vals[0] - xi)
            # For intermediate categories: P(Y=k) = F(b_{k+1} - xi) - F(b_k - xi)
            for k in range(1, K - 1):
                probs[i, k] = logistic_cdf(b_vals[k] - xi) - logistic_cdf(b_vals[k - 1] - xi)
            # Last category probability: P(Y=K-1) = 1 - F(b_{K-1} - xi)
            probs[i, K - 1] = 1.0 - logistic_cdf(b_vals[K - 2] - xi)
        return probs
