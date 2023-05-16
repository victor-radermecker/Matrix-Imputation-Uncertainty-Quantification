import numpy as np
import pandas as pd
import random
from scipy.stats import t
from tqdm import tqdm
from scipy import stats
import numba
from sklearn.preprocessing import OneHotEncoder
import matplotlib.colors as mcolors

from src.utils import *
from src.mit_cis import *


class MatrixCompletionFast:
    def __init__(
        self,
        data,
        factors,
        n_jobs=-1,
        random_seed=1601,
    ):
        self.data = data
        self.back_data = data
        self.np_data = data.values
        self.factors = factors
        self.encoder = None  # used to one-hot encode categorical variables
        self.n_jobs = n_jobs
        self.random_seed = random_seed

    # ----------------------------------------------------------------
    #               Categorical Variables Preprocessing              #
    # ----------------------------------------------------------------

    def one_hot_encode(self):
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

    # def one_hot_encode(self):
    #     print("One hot encoding the categorical features...\n")
    #     print("Initial number of features: ", self.data.shape[1], "\n")

    #     categorical_columns = self.data.select_dtypes(
    #         include=["object", "category"]
    #     ).columns
    #     self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    #     encoded_df = pd.DataFrame(
    #         self.encoder.fit_transform(self.data[categorical_columns])
    #     )
    #     encoded_df.index = self.data.index
    #     encoded_df.columns = self.encoder.get_feature_names_out(categorical_columns)

    #     non_categorical_df = self.data.drop(categorical_columns, axis=1)
    #     combined_df = pd.concat([non_categorical_df, encoded_df], axis=1)

    #     print("Final number of features: ", combined_df.shape[1], "\n")
    #     self.data_not_encoded = self.data
    #     # self.data = combined_df
    #     # self.np_data = combined_df.values
    #     return combined_df

    # def inverse_one_hot_encode(self, array):
    #     categorical_columns = self.data_not_encoded.select_dtypes(
    #         include=["object", "category"]
    #     ).columns
    #     encoded_columns = self.encoder.get_feature_names_out(categorical_columns)
    #     encoded_df = pd.DataFrame(
    #         array[:, -len(encoded_columns) :], columns=encoded_columns
    #     )

    #     # Round the imputed values before decoding
    #     encoded_df = encoded_df.round()

    #     decoded_data = self.encoder.inverse_transform(encoded_df)
    #     decoded_df = pd.DataFrame(
    #         decoded_data, columns=categorical_columns, index=self.data_not_encoded.index
    #     )

    #     non_categorical_df = pd.DataFrame(
    #         array[:, : -len(encoded_columns)],
    #         columns=self.data_not_encoded.columns.difference(categorical_columns),
    #         index=self.data_not_encoded.index,
    #     )
    #     combined_df = pd.concat([non_categorical_df, decoded_df], axis=1)

    #     return combined_df

    # ----------------------------------------------------------------
    #                       Matrix Completion                        #
    # ----------------------------------------------------------------

    def impute_dataframe(self, keep_missing_only=True):
        """
        Imputes missing values in the input dataframe using hard imputation.

        Returns:
        pandas.DataFrame: The imputed dataframe.
        """

        # if contains_cat:
        #     M = self.one_hot_encode()
        # else:
        #     M = self.np_data

        Md = self.impute_matrix()

        # M = self.np_data
        # O = self.extract_O(M)
        # Ω = self.extract_Omega(M)
        # Md = self.hard_impute(O, Ω, self.factors)

        # if contains_cat:
        #     # print the number of NaN values in Md
        #     print("Number of NaN values in Md: ", np.isnan(Md).sum())
        #     Md = self.inverse_one_hot_encode(
        #         Md,
        #     )
        #     M = self.inverse_one_hot_encode(
        #         M,
        #     )

        if keep_missing_only:
            df = pd.DataFrame(
                self.np_data, index=self.data.index, columns=self.data.columns
            )
            df = df.where(~np.isnan(df), Md)
        else:
            df = pd.DataFrame(Md, index=self.data.index, columns=self.data.columns)

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

    # Add other helper functions and methods here, such as split_data(), impute(), etc.
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

    def confidence_intervals(self, conf_lvls, train_prop, nbr_batches, val_prop, beta):

        self.conf_lvls = conf_lvls
        self.beta = beta
        self.train_proportion = train_prop
        self.nbr_batches = nbr_batches
        self.valid_proportion = val_prop
        self.val_idx, self.train_idx = self.split_data()

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
            for i in range(len(self.data.columns))
        ]
        print(len(titles))

        axes = grid.axes.flatten()
        for i in range(len(titles)-3):
            axes[i].set_title(titles[i])
            axes[i].set_ylim(0, 1)

        grid.map(sns.lineplot, "cov", "length")
        grid.add_legend(title="Method used")
        plt.show()


    def get_cis(self, alpha, method):

        conf_levels = [alpha]
        self.train_idx = np.argwhere(
            ~np.isnan(self.np_data)
        )  # we train on all data points
        miss_idx = np.argwhere(np.isnan(self.np_data))
        n_miss_idx = len(miss_idx)
        
        print("Imputing full dataset...")
        self.Md = self.impute_matrix()

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


"""
    def confidence_intervals_fast(self):

        masked_data = mask_matrix(self.np_data, self.train_idx)
        baseline_mean = np.nanmean(masked_data, axis=0)
        baseline_std = np.nanstd(masked_data, axis=0)

        O = self.extract_O(self.np_data)
        Ω = self.extract_Omega(self.np_data)
        Md = self.hard_impute(O, Ω, r=self.factors)  # full imputation

        print("Running imputation on all training batches...")
        batches = self.generate_batches()
        imputation_results = np.empty((self.nbr_batches, len(self.val_idx)))
        for i, batch in enumerate(tqdm(batches)):
            res = self.process_sample(self.val_idx, batch)
            imputation_results[i, :] = res
        print("Imputation on training data finished. \n")

        # Initialize Final Output
        columns = [
            "val_pair",
            "feature",
            "truth",
            "imputed",
            "alpha",
            "low",
            "mean",
            "up",
            "length",
            "cov",
            "std",
        ]

        val_res_list = []
        val_res_baseline_list = []
        val_res_mit_list = []

        ci_values = np.empty((len(self.conf_lvls), len(self.val_idx), 4))
        ci_values_baseline = np.empty((len(self.conf_lvls), len(self.val_idx), 4))
        ci_values_mit = np.empty((len(self.conf_lvls), len(self.val_idx), 4))

        print("Computing confidence intervals for all validation points...")
        # Precompute these values
        mean_points = np.mean(imputation_results, axis=0)
        std_points = np.std(imputation_results, axis=0)

        # Precompute MIT standard deviations
        mit_std_array = np.empty((len(self.val_idx)))
        Ω = self.extract_Omega(self.np_data)
        X = self.np_data.copy()
        X[np.isnan(X)] = 0
        r = 9
        p_observe = 1 - np.isnan(self.np_data).sum() / self.np_data.size
        sigmaS = compute_Sigma_adaptive(Md, (X - Md) * Ω, r, p_observe)
        for j, pair in enumerate(self.val_idx):
            mit_std_array[j] = np.sqrt(sigmaS[pair[0], pair[1]])

        # Get the number of val_idx and conf_lvls
        n_val_idx = len(self.val_idx)

        # Create 1D arrays to store the imputed_point, baseline_mean, and baseline_std values
        imputed_points = np.empty(n_val_idx)
        baseline_mean_array = np.empty(n_val_idx)
        baseline_std_array = np.empty(n_val_idx)
        non_nan_count = np.empty(n_val_idx)
        non_nan_count_cols = np.sum(~np.isnan(self.np_data), axis=0)

        # Fill the 1D arrays with the appropriate values
        for j, pair in enumerate(self.val_idx):
            imputed_points[j] = Md[pair[0], pair[1]]
            baseline_mean_array[j] = baseline_mean[pair[1]]
            baseline_std_array[j] = baseline_std[pair[1]]
            non_nan_count[j] = non_nan_count_cols[pair[1]]

        # Compute the ci_values and ci_values_baseline in a vectorized manner
        for i, alpha in enumerate(tqdm(self.conf_lvls)):
            ci_values[i, :, :] = np.column_stack(
                student_cis_centered(
                    mean_points, std_points, imputed_points, self.nbr_batches, alpha
                )
            )
            ci_values_baseline[i, :, :] = np.column_stack(
                student_cis_centered(
                    baseline_mean_array,
                    baseline_std_array,
                    imputed_points,
                    non_nan_count,
                    alpha,
                )
            )
            ci_values_mit[i, :, :] = np.column_stack(
                student_cis_centered(
                    baseline_mean_array,
                    mit_std_array,
                    imputed_points,
                    non_nan_count,
                    alpha,
                )
            )

        print("Confidence intervals computed. \n")

        # Loop over validation points and compute coverage for all alpha values
        print("Computing coverage for all validation points...")
        for i, pair in enumerate(tqdm(self.val_idx)):

            true_point = self.np_data[pair[0], pair[1]]
            imputed_point = Md[pair[0], pair[1]]

            # Algorithm V2
            points = imputation_results[:, i]
            cov_flags = (ci_values[:, i, 0] <= true_point) & (
                true_point <= ci_values[:, i, 2]
            )

            # Baseline
            cov_flags_baseline = (ci_values_baseline[:, i, 0] <= true_point) & (
                true_point <= ci_values_baseline[:, i, 2]
            )

            # MIT
            cov_flags_mit = (ci_values_mit[:, i, 0] <= true_point) & (
                true_point <= ci_values_mit[:, i, 2]
            )

            # Append results to list
            for j, alpha in enumerate(self.conf_lvls):
                val_res_list.append(
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
                        np.std(points),
                    ]
                )

                val_res_baseline_list.append(
                    [
                        pair,
                        pair[1],
                        true_point,
                        imputed_point,
                        alpha,
                        ci_values_baseline[j, i, 0],
                        ci_values_baseline[j, i, 1],
                        ci_values_baseline[j, i, 2],
                        ci_values_baseline[j, i, 3],
                        cov_flags_baseline[j],
                        baseline_std[pair[1]],
                    ]
                )
                val_res_mit_list.append(
                    [
                        pair,
                        pair[1],
                        true_point,
                        imputed_point,
                        alpha,
                        ci_values_mit[j, i, 0],
                        ci_values_mit[j, i, 1],
                        ci_values_mit[j, i, 2],
                        ci_values_mit[j, i, 3],
                        cov_flags_mit[j],
                        0,
                    ]
                )
        print("Coverage computed. \n")

        # Convert list to DataFrame
        self.bootstrap_results = pd.DataFrame(val_res_list, columns=self.columns)
        self.baseline_results = pd.DataFrame(
            val_res_baseline_list, columns=self.columns
        )
        self.mit_results = pd.DataFrame(val_res_mit_list, columns=self.columns)

        print("Success. Exit.")
"""
