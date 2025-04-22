import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solution import MultinomialLogReg, OrdinalLogReg  # Import our custom regression models from solution.py
from sklearn.metrics import accuracy_score  # For calculating accuracy on predictions


# ------------------------------
# Helper Functions for Bootstrapping & Coefficient Extraction
# ------------------------------

def bootstrap_estimates(model_class, X, y, n_bootstrap=1, random_state=42):
    """
    Perform bootstrapping to estimate standard errors of the model parameters.

    For each bootstrap iteration:
      - Resample the data with replacement.
      - Fit the specified model on the bootstrap sample.
      - Collect the estimated model parameters.
    Finally, compute the standard deviation of the parameters across iterations.

    Parameters:
      model_class (class): The regression model class (e.g., MultinomialLogReg or OrdinalLogReg)
      X (np.array): Feature matrix of shape (N, d)
      y (np.array): Target vector of shape (N,)
      n_bootstrap (int): Number of bootstrap iterations to run
      random_state (int): Seed for reproducibility

    Returns:
      se (np.array): Array of standard errors computed across bootstrap iterations.
    """
    # Set the seed for reproducibility
    np.random.seed(random_state)
    # Determine the number of samples in the dataset
    n_samples = X.shape[0]
    # Initialize a list to store the parameters from each bootstrap sample
    params_list = []

    # Loop over the number of bootstrap iterations
    for i in range(n_bootstrap):
        # Print progress every 10 iterations (if applicable)
        if (i + 1) % 10 == 0:
            print(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
        # Generate random indices for sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Create the bootstrap sample for features and targets
        X_boot = X[indices]
        y_boot = y[indices]
        # Fit the model on the bootstrap sample using the build() method
        model = model_class().build(X_boot, y_boot)
        # Append the estimated parameters to our list
        params_list.append(model.params_)

    # Convert the list of parameters into a numpy array for easier manipulation
    params_array = np.array(params_list)
    # Compute the standard deviation along the bootstrap iterations axis (axis=0)
    se = np.std(params_array, axis=0)
    return se


def extract_multinomial_coefficients(model, feature_names, shot_type_map):
    """
    Extract the coefficient matrix from a fitted MultinomialLogReg model.

    Parameters:
      model: A fitted instance of MultinomialLogReg
      feature_names (list): List of names of the features (e.g., ['Angle', 'Distance', ...])
      shot_type_map (dict): Mapping from shot type names to integer labels

    Returns:
      df (pd.DataFrame): A DataFrame where rows are features and columns are shot types.
                         For the baseline class, coefficients are set to 0.
    """
    # Determine number of features (d)
    d = len(feature_names)
    # Determine the number of shot types (K)
    K = len(shot_type_map)
    # Reshape the parameter vector to a matrix of shape (d, K-1)
    W = model.params_.reshape(d, K - 1)
    # For the baseline class, add a column of zeros (since we do not optimize for it)
    coef_full = np.hstack([W, np.zeros((d, 1))])
    # Order the shot type names by their numeric label in shot_type_map
    ordered_types = [stype for stype, idx in sorted(shot_type_map.items(), key=lambda item: item[1])]
    # Create a DataFrame for easy viewing
    df = pd.DataFrame(coef_full, index=feature_names, columns=ordered_types)
    return df


def extract_multinomial_standard_errors(se_array, feature_names, shot_type_map):
    """
    Reshape the bootstrap standard errors to match the coefficient matrix layout.

    Parameters:
      se_array (np.array): The array of standard errors computed from bootstrapping.
      feature_names (list): List of feature names.
      shot_type_map (dict): Mapping from shot type names to integer labels.

    Returns:
      df (pd.DataFrame): DataFrame of standard errors with the same layout as the coefficient matrix.
    """
    d = len(feature_names)
    K = len(shot_type_map)
    # Reshape the standard error array to (d, K-1)
    se_mat = se_array.reshape(d, K - 1)
    # Add a column of zeros for the baseline class
    se_full = np.hstack([se_mat, np.zeros((d, 1))])
    ordered_types = [stype for stype, idx in sorted(shot_type_map.items(), key=lambda item: item[1])]
    df = pd.DataFrame(se_full, index=feature_names, columns=ordered_types)
    return df


def plot_coefficients(coef_df, se_df, title, filename):
    """
    Plot a grouped bar chart of coefficient estimates with error bars.

    Parameters:
      coef_df (pd.DataFrame): DataFrame containing coefficient estimates.
      se_df (pd.DataFrame): DataFrame containing standard errors for the coefficients.
      title (str): Title of the plot.
      filename (str): Filename to save the plot as an SVG.
    """
    # Plot the bar chart with error bars; capsize sets the size of the error bar caps
    ax = coef_df.plot.bar(yerr=se_df, capsize=4, figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Coefficient Estimate")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(filename, format="svg")  # Save the plot as an SVG file
    plt.close()  # Close the plot to free memory


def plot_ordinal_parameters(ordinal_model, se_array, feature_names, title, filename_prefix):
    """
    Plot the beta coefficients and threshold parameters from an OrdinalLogReg model,
    with error bars from bootstrapping. Saves two SVG files.

    Parameters:
      ordinal_model: A fitted OrdinalLogReg model.
      se_array (np.array): Array of standard errors from bootstrapping.
      feature_names (list): List of feature names for the beta coefficients.
      title (str): Base title for the plots.
      filename_prefix (str): Prefix for the filenames of the saved SVGs.
    """
    d = len(feature_names)
    # Extract all parameters from the ordinal model.
    params = ordinal_model.params_
    # The first d elements are the beta coefficients.
    beta = params[:d]
    # The remaining parameters correspond to the thresholds.
    threshold_params = params[d:]

    # Plot beta coefficients.
    plt.figure(figsize=(8, 6))
    plt.bar(feature_names, beta, yerr=se_array[:d], capsize=4)
    plt.title(title + " - Beta Coefficients")
    plt.ylabel("Beta Estimate")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Save the beta coefficients plot as an SVG.
    plt.savefig(f"{filename_prefix}_beta.svg", format="svg")
    plt.close()

    # Plot threshold parameters.
    indices = [f"Threshold {i + 1}" for i in range(len(threshold_params))]
    plt.figure(figsize=(6, 4))
    plt.bar(indices, threshold_params, yerr=se_array[d:], capsize=4)
    plt.title(title + " - Threshold Parameters")
    plt.ylabel("Threshold Estimate")
    plt.tight_layout()
    # Save the threshold parameters plot as an SVG.
    plt.savefig(f"{filename_prefix}_thresholds.svg", format="svg")
    plt.close()


def load_real_data(dataset_path='dataset.csv'):
    """
    Load and preprocess the real basketball shot dataset.

    The function reads the CSV file, maps the shot types to numeric labels,
    one-hot encodes categorical predictors, and returns the feature matrix,
    target vector, list of feature names, and shot type mapping.

    Parameters:
      dataset_path (str): Path to the CSV file.

    Returns:
      X (np.array): Preprocessed feature matrix.
      y (np.array): Numeric target vector.
      feature_columns (list): Names of the features.
      shot_type_map (dict): Mapping of shot type names to numeric labels.
    """
    # Read the CSV file using ';' as the separator
    data = pd.read_csv(dataset_path, sep=';')
    # Get the unique shot types and create a mapping to integers
    shot_types = data['ShotType'].unique()
    shot_type_map = {stype: i for i, stype in enumerate(shot_types)}
    # Map the shot types to numeric labels in the DataFrame
    data['ShotType_num'] = data['ShotType'].map(shot_type_map)
    # Define categorical features that need one-hot encoding
    categorical_features = ['Competition', 'PlayerType', 'Transition', 'TwoLegged', 'Movement']
    # One-hot encode the categorical features; drop the first level to avoid multicollinearity
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    # Define feature columns as all columns except the original shot type and its numeric version
    feature_columns = [col for col in data_encoded.columns if col not in ['ShotType', 'ShotType_num']]
    # Extract feature matrix and target vector from the DataFrame
    X = data_encoded[feature_columns].values
    y = data_encoded['ShotType_num'].values
    return X, y, feature_columns, shot_type_map


# ------------------------------
# Part 2.1: Application of Multinomial Regression on Real Data
# ------------------------------

def application_multinomial_regression(dataset_path='dataset.csv'):
    """
    Load the basketball shot dataset, fit the multinomial logistic regression model,
    and print class distribution information.

    Parameters:
      dataset_path (str): Path to the dataset CSV file.

    Returns:
      model: Fitted MultinomialLogReg model.
      X: Preprocessed feature matrix.
      y: Numeric target vector.
      feature_columns: List of feature names.
      shot_type_map: Mapping of shot type names to numeric labels.
    """
    # Load and preprocess the dataset.
    X, y, feature_columns, shot_type_map = load_real_data(dataset_path)
    print("Dataset shape:", X.shape)
    print("Features:", feature_columns)
    print("Shot type mapping:", shot_type_map)

    # Build the multinomial logistic regression model.
    model = MultinomialLogReg().build(X, y)
    # Use the fitted model to predict probabilities on the training data.
    probs = model.predict(X)
    # Choose the class with the highest probability for each sample.
    predicted_classes = np.argmax(probs, axis=1)

    # Print the distribution of predicted and actual classes.
    print("Predicted class distribution (counts):", np.bincount(predicted_classes))
    print("Actual class distribution (counts):", np.bincount(y))
    return model, X, y, feature_columns, shot_type_map


# ------------------------------
# Part 2.2: Synthetic DGP for Ordinal Regression
# ------------------------------

def multinomial_bad_ordinal_good(n_samples=1000, random_state=42):
    """
    Generate synthetic data with an ordinal outcome.

    For each observation:
      - Sample features from a standard normal distribution.
      - Compute a latent score using a fixed coefficient vector and added noise.
      - Apply thresholds t1 = -0.5 and t2 = 0.5 to create three ordered categories (0, 1, 2).

    Parameters:
      n_samples (int): Number of observations to generate.
      random_state (int): Seed for reproducibility.

    Returns:
      X (np.array): Generated feature matrix.
      y (np.array): Generated ordinal target vector.
    """
    # Set the seed for reproducibility.
    np.random.seed(random_state)
    d = 3  # Number of features
    # Generate n_samples observations with d features from a standard normal distribution.
    X = np.random.randn(n_samples, d)
    # Define the true coefficients for the latent variable.
    beta = np.array([1.0, -2.0, 0.5])
    # Compute the latent variable (eta) with added Gaussian noise.
    eta = X.dot(beta) + 0.5 * np.random.randn(n_samples)
    # Define thresholds for the ordinal outcome.
    t1, t2 = -0.5, 0.5
    # Initialize target vector with zeros.
    y = np.zeros(n_samples, dtype=int)
    # Assign category 1 to samples where eta is greater than or equal to t1.
    y[eta >= t1] = 1
    # Assign category 2 to samples where eta is greater than or equal to t2.
    y[eta >= t2] = 2
    return X, y


# ------------------------------
# Main Execution Block
# ------------------------------

if __name__ == "__main__":
    # ----- Part 2.1: Multinomial Regression on Real Data -----
    print("=== Part 2.1: Application of Multinomial Regression ===")
    try:
        # Load and fit the multinomial regression model on the real dataset.
        multinom_model, X_real, y_real, feature_columns, shot_type_map = application_multinomial_regression(
            'dataset.csv')
    except Exception as e:
        # Print error message and exit if the dataset cannot be loaded or processed.
        print("Error loading or processing dataset.csv. Please ensure the file exists and is formatted correctly.")
        print("Exception:", e)
        exit(1)

    # Compute bootstrap standard errors for the multinomial model using the real data.
    se_real = bootstrap_estimates(MultinomialLogReg, X_real, y_real, n_bootstrap=1)

    # Extract the coefficient matrix as a DataFrame.
    coef_df = extract_multinomial_coefficients(multinom_model, feature_columns, shot_type_map)
    print("Coefficient matrix:")
    print(coef_df)

    # Extract the standard errors in a matching DataFrame.
    se_df = extract_multinomial_standard_errors(se_real, feature_columns, shot_type_map)
    print("Bootstrap standard errors:")
    print(se_df)

    # Plot and save the coefficient estimates with error bars as an SVG file.
    plot_coefficients(coef_df, se_df,
                      title="Multinomial Regression Coefficients (Real Data)",
                      filename="multinomial_coefficients.svg")
    print("Coefficient plot saved as 'multinomial_coefficients.svg'")

    # (Interpretation Example)
    # One might hypothesize, for example, that if 'Competition_NBA' has a positive coefficient for 'layup',
    # then NBA games might favor layups due to factors like player attributes or defensive strategies.

    # ----- Part 2.2: Synthetic DGP for Ordinal Regression -----
    print("\n=== Part 2.2: Application of Ordinal Regression on a Synthetic DGP ===")
    # Generate synthetic ordinal data using the defined DGP.
    X_dgp, y_dgp = multinomial_bad_ordinal_good(n_samples=1000)
    # Fit a multinomial regression model on the synthetic data.
    model_multi = MultinomialLogReg().build(X_dgp, y_dgp)
    # Fit an ordinal regression model on the synthetic data.
    model_ord = OrdinalLogReg().build(X_dgp, y_dgp)

    # Predict on the synthetic data using both models.
    preds_multi = np.argmax(model_multi.predict(X_dgp), axis=1)
    preds_ord = np.argmax(model_ord.predict(X_dgp), axis=1)

    # Compute accuracy for both models.
    accuracy_multi = accuracy_score(y_dgp, preds_multi)
    accuracy_ord = accuracy_score(y_dgp, preds_ord)
    print("Synthetic DGP - Multinomial Model Accuracy: {:.2f}%".format(accuracy_multi * 100))
    print("Synthetic DGP - Ordinal Model Accuracy: {:.2f}%".format(accuracy_ord * 100))
