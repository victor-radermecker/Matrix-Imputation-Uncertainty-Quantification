import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import *
from src.mit_cis import *
from sklearn.metrics import r2_score


class MatrixCompletionFast:
    def __init__(
        self,
        data,
        factors,
        val_prop,
        n_jobs=-1,
        random_seed=1601,
    ):
        self.data = data
        self.back_data = data
        self.np_data = data.values
        self.factors = factors
        self.encoder = None
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Generate validation dataset
        self.valid_proportion = val_prop
        self.val_idx, self.train_idx = self.split_data()

        # Find the optimal number of factors
        if isinstance(self.factors, list):
            print("Range of factors specified. Optimizing factors...")
            self.factors = self.optimize_factors()

    # ----------------------------------------------------------------
    #               Categorical Variables Preprocessing              #
    # ----------------------------------------------------------------

    def one_hot_encode(self):
        """
        One-hot encode the categorical columns of the dataset, storing the result in
        self.data and self.np_data. A mask is applied to the encoded columns to
        maintain the original NaN values.
        """
        self.cat_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns
        encoded_df = pd.get_dummies(self.data, columns=self.cat_columns, dummy_na=False)

        # Create a mask to identify NaN values in the original categorical columns
        nan_mask = self.data[self.cat_columns].isna()

        # Apply the mask to the encoded columns
        for col in self.cat_columns:
            col_prefix = col + "_"
            matching_cols = [c for c in encoded_df.columns if c.startswith(col_prefix)]
            encoded_df.loc[nan_mask[col], matching_cols] = np.nan

        self.data = encoded_df
        self.np_data = encoded_df.values

    def inverse_one_hot_encode(self, imputed_matrix):
        """
        Inverse one-hot encoding of the imputed matrix, converting the encoded columns
        back to their original categorical format.

        Parameters:
        -----------
        imputed_matrix : np.ndarray
            The imputed matrix with one-hot encoded categorical columns.

        Returns:
        --------
        decoded_df : pd.DataFrame
            The imputed DataFrame with the original categorical columns restored.
        """
        decoded_df = pd.DataFrame(
            imputed_matrix, columns=self.data.columns, index=self.data.index
        )
        for col in self.cat_columns:
            col_prefix = col + "_"
            matching_cols = [c for c in decoded_df.columns if c.startswith(col_prefix)]
            decoded_values = (
                decoded_df[matching_cols].idxmax(axis=1).str.replace(col_prefix, "")
            )
            decoded_df[col] = decoded_values
            decoded_df.drop(matching_cols, axis=1, inplace=True)

        decoded_df = decoded_df[self.back_data.columns]

        return decoded_df

    # ----------------------------------------------------------------
    #                      Factors Optimization                      #
    # ----------------------------------------------------------------

    def compute_r2_on_validation(
        self,
        original_validation_df,
        imputed_validation_df,
    ):
        """
        Compute R^2 scores for each feature in the validation set.

        Parameters:
        -----------
        original_validation_df : pd.DataFrame
            DataFrame containing the original validation values.
        imputed_validation_df : pd.DataFrame
            DataFrame containing the imputed values for the validation set.

        Returns:
        --------
        r2_scores : dict
            A dictionary containing the R^2 score for each feature.
        """
        r2_scores = {}
        for col in original_validation_df.columns:
            original_values = original_validation_df[col].dropna()
            imputed_values = imputed_validation_df.loc[original_values.index, col]
            r2 = r2_score(original_values, imputed_values)
            r2_scores[col] = r2
        return r2_scores

    def optimize_factors(self):
        """
        Optimize the number of factors for matrix completion by computing the
        average R^2 score for each factor value in self.factors.

        This method updates the self.factors attribute to the optimal number
        of factors and stores the R^2 scores for all features for all factor
        values in self.r2_scores_df.
        """
        average_r2_scores = []

        # Create a DataFrame to store R^2 scores for all features for all factor values
        r2_scores_df = pd.DataFrame(
            columns=[f"Factor: {factor}" for factor in self.factors],
            index=self.data.columns,
            dtype=float,
        )

        # Create a copy of the data to hide the validation values
        masked_data = mask_matrix(self.np_data, self.train_idx)

        for factor in self.factors:
            imputed_data_matrix = self.impute(masked_data, factor)
            imputed_data = pd.DataFrame(
                imputed_data_matrix, index=self.data.index, columns=self.data.columns
            )
            val_data_matrix = mask_matrix(self.np_data, self.val_idx)
            val_data = pd.DataFrame(
                val_data_matrix, index=self.data.index, columns=self.data.columns
            )
            r2_scores = self.compute_r2_on_validation(
                val_data, imputed_data
            )
            avg_r2 = sum(r2_scores.values()) / len(r2_scores)
            average_r2_scores.append(avg_r2)

            # Store R^2 scores for the current factor in the r2_scores_df DataFrame
            r2_scores_df.loc[:, f"Factor: {factor}"] = pd.Series(r2_scores)

            print(f"Average R^2 score for {factor} factors: {avg_r2:.4f}")

        best_factors = self.factors[np.argmax(average_r2_scores)]
        self.factors = best_factors
        self.r2_scores_df = r2_scores_df
        print(
            f"The optimal number of factors is {best_factors}. \n\nNew factor successfully saved."
        )

    # ----------------------------------------------------------------
    #                       Matrix Completion                        #
    # ----------------------------------------------------------------

    def impute_dataframe(self, keep_missing_only=True):
        """
        Imputes missing values in the input dataframe using hard imputation.

        Returns:
        pandas.DataFrame: The imputed dataframe.
        """

        self.Md = self.impute_matrix()

        if keep_missing_only:
            df = pd.DataFrame(
                self.np_data, index=self.data.index, columns=self.data.columns
            )
            df = df.where(~np.isnan(df), self.Md)
        else:
            df = pd.DataFrame(self.Md, index=self.data.index, columns=self.data.columns)

        return df

    def impute_matrix(self):
        """
        Imputes missing values in the input matrix using hard imputation.

        Returns:
        ndarray: The imputed matrix.
        """
        O = self.extract_O(self.np_data)
        Ω = self.extract_Omega(self.np_data)
        Md = self.hard_impute(O, Ω, r=self.factors)
        return Md

    def impute(self, M, r):
        """
        Imputes missing values in a given matrix using hard imputation.

        Parameters:
        M (ndarray): The input matrix.
        r (int): The number of factors to use for imputation.

        Returns:
        ndarray: The imputed matrix.
        """
        O = self.extract_O(M)
        Ω = self.extract_Omega(M)
        Md = self.hard_impute(O, Ω, r)
        return Md

    def extract_O(self, M):
        """
        Extracts the observed entries of a given matrix by replacing missing values with zero.

        Parameters:
        M (ndarray): The input matrix.

        Returns:
        ndarray: The observed entries of the input matrix with missing values replaced by zero.
        """
        O = np.copy(M)
        O[np.isnan(O)] = 0
        return O

    def extract_Omega(self, M):
        """
        Extracts the binary mask of observed entries of a given matrix.

        Parameters:
        M (ndarray): The input matrix.

        Returns:
        ndarray: A binary mask of the observed entries of the input matrix, where a value of 1 denotes an observed entry and a value of 0 denotes a missing entry.
        """
        Omega = np.zeros(M.shape, dtype=int)
        Omega[~np.isnan(M)] = 1
        return Omega

    def hard_impute(self, O, Ω, r=1, tolerance=1e-3, max_iter=3000):
        """
        Function to perform the matrix completion.

        Args:
            O: 2D numpy array, missing values replaced by 0's and non-na's are original values
            Ω: 2D numpy arry, binary matrix with 0 where the matrix has missing values and 1 otherwise
            r: int, number of factors used in the imputation
            tolerance: float, tolerance value for managing convergence
            max_iter: int, maximum number of iterations allowed

        Returns:
            M: 2D numpy array of the completed matrix in the original dimensions
        """
        M = np.zeros_like(O)
        for T in range(max_iter):
            M_new = self.SVD(O * Ω + (1 - Ω) * M, r)
            # print('Iteration:', T, end = "\r")
            if np.linalg.norm(M - M_new) < np.linalg.norm(M) * tolerance:
                break
            M = M_new
        return M

    def SVD(self, M, r):
        """
        Function to perfrom singluar value decomposition for the matrix completion. Uses the
        first r singular values to reconstruct the factorised matrix

        Args:
            M: numpy 2D array, matrix to be factorised
            r: number of singular values to use

        Returns:
            M_fact: 2D numpy array of the reconstructed matrix in the original dimensions
        """
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        s[r:] = 0
        return (u * s).dot(vh)

    # ----------------------------------------------------------------
    #                Confidence Intervals Generation                 #
    # ----------------------------------------------------------------

    def split_data(self):
        """
        Splits the input data into training and validation sets using the specified proportion.

        Returns:
        tuple: A tuple containing the indices of the validation set and training set.
        """
        val_idx, train_idx = a_b_split(self.np_data, self.valid_proportion)
        summ = split_summary(self.np_data, val_idx, train_idx)
        print(summ.head(5), "\n")
        summary = summ.sum()

        print("Number of validation points:", summary["val"])
        print("Number of training points:", summary["train"])
        print("Number of missing points:", summary["miss"], "\n")

        return val_idx, train_idx

    def generate_batches(self):
        """
        Generates batches of masked training data for matrix imputation.

        Returns:
        list: A list of batches, where each batch is a masked version of the input data.
        """
        batches = []
        for i in range(self.nbr_batches):
            batch_idx = training_sample(
                self.np_data, self.train_idx, self.train_proportion
            )
            masked_data = mask_matrix(self.data, batch_idx)
            batches.append(masked_data)
        return batches

    # define the function to process each batch
    def process_sample(self, idx, batch):
        Md = self.impute(batch, self.factors)
        return Md[idx[:, 0], idx[:, 1]]

    def student_cis_centered(self, mean, std, center, nbr_points, confidence_level):
        """
        Calculates the centered confidence interval for a given sample mean and standard deviation using the Student's t-distribution.

        Parameters:
        mean (float): The sample mean.
        std (float): The sample standard deviation.
        center (float): The center point for the confidence interval.
        nbr_points (int): The number of data points used to calculate the mean and standard deviation.
        confidence_level (float): The confidence level for the interval, expressed as a decimal.

        Returns:
        tuple: A tuple containing the lower bound, center point, upper bound, and length of the confidence interval.
        """
        bounds = st.t.interval(
            confidence_level, nbr_points - 1, loc=mean, scale=std / np.sqrt(nbr_points)
        )
        length = bounds[1] - bounds[0]
        lower_bound, upper_bound = center - length / 2, center + length / 2
        return lower_bound, center, upper_bound, length

    def compute_mean_imputation_ci(
        self, mean, std, center, nbr_points, confidence_level, method
    ):
        """
        Compute the confidence interval for the mean imputed value in a column of a DataFrame using the t-distribution.

        Args:
        - df: pandas DataFrame containing the data
        - col_name: string, the name of the column to compute the confidence interval for
        - confidence_level: float between 0 and 1, the desired confidence level

        Returns:
        - tuple of four floats: lower bound, center, upper bound, and length of the confidence interval
        """
        # Compute the standard error of the mean
        se = std / np.sqrt(nbr_points)
        alpha = 1 - confidence_level

        if method == "t":
            t_value = st.t.ppf(1 - alpha / 2, nbr_points - 1)
            lower_bound = center - t_value * se * self.beta
            upper_bound = center + t_value * se * self.beta

        elif method == "normal":
            z_score = st.norm.ppf(1 - alpha / 2)
            lower_bound = center - z_score * se * self.beta
            upper_bound = center + z_score * se * self.beta

        length = upper_bound - lower_bound
        return lower_bound, center, upper_bound, length

    def baseline_confidence_intervals(self, conf_lvls, idx):
        """
        Computes confidence intervals for the baseline mean imputation method.

        Parameters:
        conf_lvls (list): A list of confidence levels expressed as decimals.
        idx (list): A list of indices to compute confidence intervals for.

        Returns:
        ndarray: An array of confidence interval values for each confidence level and index.
        """
        ci_values_baseline = np.empty((len(conf_lvls), len(idx), 4))

        # Compute the ci_values and ci_values_baseline in a vectorized manner
        for i, alpha in enumerate(tqdm(conf_lvls)):
            ci_values_baseline[i, :, :] = np.column_stack(
                self.compute_mean_imputation_ci(
                    self.baseline_mean_array,
                    self.baseline_std_array,
                    self.baseline_mean_array,
                    self.non_nan_count,
                    alpha,
                    method="normal",
                )
            )

        return ci_values_baseline

    def bootstrap_confidence_intervals(self, conf_lvls, idx):
        # Precompute these values
        ci_values_bootstrap = np.empty((len(conf_lvls), len(idx), 4))
        mean_points = np.mean(self.imputation_results, axis=0)
        std_points = np.std(self.imputation_results, axis=0)

        # Compute the ci_values and ci_values_baseline in a vectorized manner
        for i, alpha in enumerate(tqdm(conf_lvls)):
            ci_values_bootstrap[i, :, :] = np.column_stack(
                self.compute_mean_imputation_ci(
                    mean_points,
                    std_points,
                    self.imputed_points,
                    self.non_nan_count,
                    alpha,
                    method="t",
                )
            )

        return ci_values_bootstrap

    def init_baseline(self):
        # Get the number of val_idx and conf_lvls
        n_val_idx = len(self.val_idx)
        masked_data = mask_matrix(self.np_data, self.train_idx)

        self.baseline_mean = np.nanmean(masked_data, axis=0)
        self.baseline_std = np.nanstd(masked_data, axis=0)

        self.imputed_points = np.empty(n_val_idx)
        self.baseline_mean_array = np.empty(n_val_idx)
        self.baseline_std_array = np.empty(n_val_idx)
        self.non_nan_count = np.empty(n_val_idx)
        self.non_nan_count_cols = np.sum(~np.isnan(self.np_data), axis=0)

        # Fill the 1D arrays with the appropriate values
        for j, pair in enumerate(self.val_idx):
            self.imputed_points[j] = self.Md[pair[0], pair[1]]
            self.baseline_mean_array[j] = self.baseline_mean[pair[1]]
            self.baseline_std_array[j] = self.baseline_std[pair[1]]
            self.non_nan_count[j] = self.non_nan_count_cols[pair[1]]

    def mit_confidence_intervals(self, conf_lvls, idx):
        ci_values_mit = np.empty((len(conf_lvls), len(idx), 4))

        # Precompute MIT standard deviations
        mit_std_array = np.empty((len(idx)))
        Ω = extract_Omega(self.np_data)
        X = self.np_data.copy()
        X[np.isnan(X)] = 0
        r = 9
        p_observe = 1 - np.isnan(self.np_data).sum() / self.np_data.size
        sigmaS = compute_Sigma_adaptive(self.Md, (X - self.Md) * Ω, r, p_observe)

        for j, pair in enumerate(idx):
            mit_std_array[j] = np.sqrt(sigmaS[pair[0], pair[1]])

        for i, alpha in enumerate(tqdm(conf_lvls)):
            ci_values_mit[i, :, :] = np.column_stack(
                self.compute_mean_imputation_ci(
                    self.baseline_mean_array,
                    mit_std_array,
                    self.imputed_points,
                    self.non_nan_count,
                    alpha,
                    method="normal",
                )
            )

        return ci_values_mit

    def batch_imputations(self, idx):
        batches = self.generate_batches()
        imputation_results = np.empty((self.nbr_batches, len(idx)))
        for i, batch in enumerate(tqdm(batches)):
            res = self.process_sample(idx, batch)
            imputation_results[i, :] = res
        return imputation_results

    def format_results(self, ci_values, idx, conf_lvls):
        results = []
        for i, pair in enumerate(tqdm(idx)):
            true_point = self.np_data[pair[0], pair[1]]
            imputed_point = self.Md[pair[0], pair[1]]

            cov_flags = (ci_values[:, i, 0] <= true_point) & (
                true_point <= ci_values[:, i, 2]
            )

            for j, alpha in enumerate(conf_lvls):
                results.append(
                    [
                        pair,
                        pair[1],
                        true_point,
                        imputed_point,
                        alpha,
                        ci_values[j, i, 0],
                        ci_values[j, i, 1],
                        ci_values[j, i, 2],
                        ci_values[j, i, 3],
                        cov_flags[j],
                    ]
                )
        return results

    def confidence_intervals(self, conf_lvls, train_prop, nbr_batches, beta):
        """
        Compute confidence intervals for all validation points using baseline,
        bootstrap, and multiple imputations techniques.

        Parameters:
        -----------
        conf_lvls : list
            List of desired confidence levels.
        train_prop : float
            Proportion of the data to use for training.
        nbr_batches : int
            Number of batches to run the imputation process.
        beta : float
            Artificially inflate confidence intervals by a factor beta to reach the desired coverage. (Non-Gaussian noises)
        """
        self.conf_lvls = conf_lvls
        self.beta = beta
        self.train_proportion = train_prop
        self.nbr_batches = nbr_batches

        print("Imputing full matrix...")
        self.Md = self.impute_matrix()
        print("Imputation finished. \n")

        print("Running imputation on all training batches...")
        self.imputation_results = self.batch_imputations(self.val_idx)
        print("Imputation on training data finished. \n")

        # Initialize Final Output
        self.columns = [
            "val_pair",
            "feature",
            "truth",
            "imputed",
            "confidence_level",
            "low",
            "mean",
            "up",
            "length",
            "cov",
        ]

        self.init_baseline()

        print("Computing confidence intervals for all validation points...")
        ci_values_baseline = self.baseline_confidence_intervals(
            self.conf_lvls, self.val_idx
        )
        ci_values_bootstrap = self.bootstrap_confidence_intervals(
            self.conf_lvls, self.val_idx
        )
        ci_values_mit = self.mit_confidence_intervals(self.conf_lvls, self.val_idx)
        print("Confidence intervals computed. \n")

        # Loop over validation points and compute coverage for all alpha values
        print("Computing coverage for all validation points...")
        baseline = self.format_results(ci_values_baseline, self.val_idx, self.conf_lvls)
        bootstrap = self.format_results(
            ci_values_bootstrap, self.val_idx, self.conf_lvls
        )
        mit = self.format_results(ci_values_mit, self.val_idx, self.conf_lvls)
        print("Coverage computed. \n")

        # Convert list to DataFrame
        self.bootstrap_results = pd.DataFrame(bootstrap, columns=self.columns)
        self.baseline_results = pd.DataFrame(baseline, columns=self.columns)
        self.mit_results = pd.DataFrame(mit, columns=self.columns)

        print("Success. Exit.")

    def agg_results(self, data, method):
        out = (
            data.groupby(["feature", "confidence_level"])["length", "cov"]
            .mean()
            .reset_index()
        )
        out["method"] = method
        return out

    def summarize_results(self):
        self.baseline = self.agg_results(self.baseline_results, "Baseline")
        self.bootstrap = self.agg_results(self.bootstrap_results, "Bootstrap")
        self.mit = self.agg_results(self.mit_results, "MIT")
        return self.baseline, self.bootstrap, self.mit

    def plot_results_feature(self):
        """
        Plot the coverage versus the length of the confidence intervals for each
        feature in the dataset, comparing the results of the baseline, bootstrap,
        and multiple imputations techniques.
        """
        # Plot the coverage beased of the length
        merged_df = pd.concat([self.baseline, self.bootstrap, self.mit])
        grid = sns.FacetGrid(
            merged_df,
            col="feature",
            hue="method",
            palette=["orange", "blue", "green"],
            col_wrap=5,
        )

        titles = [
            f"Feature {i} \n Missingness: {round((self.data.isna().sum()[i]/len(self.data))*100,1)}%"
            for i in range(20)
        ]

        axes = grid.axes.flatten()
        for i in range(20):
            axes[i].set_title(titles[i])
            # axes[i].set_ylim(0, 1)

        grid.map(sns.lineplot, "cov", "length")
        grid.add_legend(title="Method used")
        plt.show()

    def get_cis(self, alpha, method):
        """
        Compute confidence intervals for the imputed values using the specified method
        (baseline, bootstrap, or multiple imputations).

        Parameters:
        -----------
        alpha : float
            The desired confidence level (e.g., 0.95 for 95% confidence intervals).
        method : str
            The method to use for computing confidence intervals. Must be one of
            'baseline', 'bootstrap', or 'mit' (multiple imputations).

        Returns:
        --------
        cis : pd.DataFrame
            A DataFrame containing the computed confidence intervals for the imputed values,
            along with the indices of the missing values and the imputed values themselves.
            The columns are: 'indices', 'low', 'up', and 'imputed'.
        """
        conf_levels = [alpha]
        # we train on all data points
        self.train_idx = np.argwhere(~np.isnan(self.np_data))
        miss_idx = np.argwhere(np.isnan(self.np_data))
        n_miss_idx = len(miss_idx)

        self.baseline_mean = np.nanmean(self.np_data, axis=0)
        self.baseline_std = np.nanstd(self.np_data, axis=0)

        self.imputed_points = np.empty(n_miss_idx)
        self.baseline_mean_array = np.empty(n_miss_idx)
        self.baseline_std_array = np.empty(n_miss_idx)
        self.non_nan_count = np.empty(n_miss_idx)
        self.non_nan_count_cols = np.sum(~np.isnan(self.np_data), axis=0)

        # Fill the 1D arrays with the appropriate values
        for j, pair in enumerate(miss_idx):
            self.imputed_points[j] = self.Md[pair[0], pair[1]]
            self.baseline_mean_array[j] = self.baseline_mean[pair[1]]
            self.baseline_std_array[j] = self.baseline_std[pair[1]]
            self.non_nan_count[j] = self.non_nan_count_cols[pair[1]]

        if method == "baseline":
            ci_values_baseline = self.baseline_confidence_intervals(
                conf_levels, miss_idx
            )
            data = self.format_results(ci_values_baseline, miss_idx, conf_levels)

        elif method == "bootstrap":
            print("Running imputation on all training batches...")
            self.imputation_results = self.batch_imputations(miss_idx)
            print("Imputation on training data finished. \n")
            ci_values_bootstrap = self.bootstrap_confidence_intervals(
                conf_levels, miss_idx
            )
            data = self.format_results(ci_values_bootstrap, miss_idx, conf_levels)

        elif method == "mit":
            ci_values_mit = self.mit_confidence_intervals(conf_levels, miss_idx)
            data = self.format_results(ci_values_mit, miss_idx, conf_levels)

        data = pd.DataFrame(data, columns=self.columns)
        cis = data[data["confidence_level"] == alpha][
            ["val_pair", "low", "up", "imputed"]
        ]
        cis.rename(columns={"val_pair": "indices"}, inplace=True)
        return cis
