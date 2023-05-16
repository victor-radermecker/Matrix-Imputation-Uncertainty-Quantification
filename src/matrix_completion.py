import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy import optimize
import math 
from random import Random
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)


class MatrixCompletion(object):
    """
    Create a class of matrix completion objects that can be used to run imputation for a data set passed.
    The class will be as generalizable as possible to imputation projects. 
    
    Args:
        df (pd.DataFrame): Input dataframe - ideally already processed with fetaures selected. 
        masking_metod: str, method of masking the data to get the missing indices matrix
        masking_proportion: float, proportion of the data to be masked, default 10%
        random_seed: Int, seed to be used throughout the imputation
        
    Example usage: 
    import matrix_completion as MC
    mat = MC.MatrixCompletion(synthetic_data, masking_method='uniform', masking_proportion = 0.2)
    
    or 
    
    mat = MC.MatrixCompletion(synthetic_data, masking_method='custom', custom_mask=missing_indices_matrix)
    mat.run_completion(6)
    """
    
    def __init__(
        self,
        df,
        masking_method = 'uniform',
        custom_mask = None,
        masking_proportion = 0.1,
        random_seed = 2022,
        threshold = 0.2
    ):
        self.data = df.copy()
        self.seed = random_seed,
        self.masking_proportion = masking_proportion
        self.threshold = threshold
        
        if masking_method == 'uniform':
            print("Creating uniform mask with seed {}. To input custom mask, input mask when initializing or use method \'set_custom_mask()'".format(self.seed))
            self.masked_copy, self.missing_indices_matrix = self.uniform_masking()
        
        if masking_method == 'pattern_overlay':
            self.masked_copy, self.missing_indices_matrix, self.pattern_summary_df = self.mask_data_patterned(self.data,
                                                                                     threshold=self.threshold,
                                                                                     masking_perc=self.masking_proportion,
                                                                                     seed=self.seed)
            print("Patterns applied. Summary:")
            print(self.pattern_summary_df)
            
        if masking_method == 'custom':
            self.masked_copy, self.missing_indices_matrix = self.set_custom_mask(custom_mask)
            
        if masking_method not in ('uniform', 'custom', 'pattern_overlay'): 
            print("Masking failed. parameter masking_method must be one of (\'uniform\', \'custom\', \'pattern_overlay\')")
            self.masked_copy, self.missing_indices_matrix = None, None
            
        self.omega_st, self.raw_st = self.get_raw_omega_mats()
        
    def get_gridsearch_mat_as_df(self):
        """
        Function to return the matrix from the gridsearch as a dataframe
        """
        imputed_df = pd.DataFrame(self.gridsearch_best_mat, columns = self.data.columns)
        return imputed_df
    
    def get_mat_as_df(self):
        """
        Function to return the matrix as a dataframe
        """
        imputed_df = pd.DataFrame(self.imputed_matrix, columns = self.data.columns)
        return imputed_df
    
    def uniform_masking(self):
        """
        Function to mask a uniform percentage of the data for the self-supervised learning aspect of the model

        Args:
            self
            
        Outputs:
            mar_copy: a uniformly masked copy of the data
            missing_indices_matrix: a binary matrix containing the indices of the masked values
        """
        myRandom = Random(self.seed)
        missing_indices_matrix = np.zeros((self.data.shape[0], self.data.shape[1]))
        missing_p = self.masking_proportion
        for i in range(self.data.shape[1]):
            idxs = myRandom.sample(range(self.data.shape[0]), int(round(missing_p * self.data.shape[0])))
            missing_indices_matrix[idxs, i] = 1
            
        mar_copy = self.data.mask(missing_indices_matrix == 1)
        return mar_copy, missing_indices_matrix        
        
        
    def set_custom_mask(self, mask):
        """
        Function to allow the user to input a custom mask pattern

        Args:
            mask: 2D array/dataframe, binary with a 1 where the entries have been masked
            
        Outputs:
            mar_copy: 2D array/dataframe, a uniformly masked copy of the data
            mask
        """    
        mar_copy = self.data.copy()
        mar_copy = mar_copy.mask(mask == 1)
        self.masked_copy = mar_copy
        return mar_copy, mask                
    
    def mask_data_patterned(self, data_to_mask, threshold=0.2, masking_perc=0.1, seed=2022):

        """
        Function to perform the patterned masking used throughout the project. Splits the data into the 
        two subsets A and B with low and high missingness respectively. 

        Args:
            data_to_mask: pd.DataFrame, dataframe which is to be masked
            threshold: float in (0, 1), missingness threshold by which to separate the two subsets
            masking_perc: float in (0, 1), the fraction of the rows to be sampled for masking
            see: int, random seed used in the masking
            
        Outputs:
            resultant_row:, array/series, resulting row2 after entries have been masked
        """  
        
        # partition based on missingness threshold 
        # sum over col / numb of col
        partition = data_to_mask.isnull().sum(axis=1) / data_to_mask.shape[1] <= threshold

        # get partition indices
        # with row idx
        A = []
        B = []
        idx = 0
        for item in partition:
            if item:
                A.append(idx)
            else:
                B.append(idx)
            idx += 1

        # get some summary stats for the partition 
        num_true = partition.sum()
        perc_true = num_true / self.data.shape[0]
        B_missing = (data_to_mask.iloc[B, :].isnull().sum(axis=1) / data_to_mask.shape[1]).mean()
        A_missing = (data_to_mask.iloc[A, :].isnull().sum(axis=1) / data_to_mask.shape[1]).mean()
        summary_df = pd.DataFrame({
            'Category': ['Subset A', 'Subset B'],
            'Size': [num_true, data_to_mask.shape[0] - num_true],
            'Percentage of Total': [perc_true, 1 - perc_true],
            'Percentage Missing':[A_missing, B_missing]
        })

        
        
        # get the indices for which to apply the masking 
        if masking_perc * data_to_mask.shape[0] > len(A) or masking_perc * data_to_mask.shape[0] > len(B):
            print("Sets too small. Consider increasing/decreasing threshold to get a more even split or decreasing masking percentage")
            return None
        
        FuncRandom = Random(2022)
        B_patterns = FuncRandom.sample(B, int(round(masking_perc * data_to_mask.shape[0])))
        A_targets = FuncRandom.sample(A, int(round(masking_perc * data_to_mask.shape[0])))
        missing_indices_matrix = self.get_patterned_missing_indices_matrix(B_patterns, A_targets)

        # get a copy which we will then mask, sampling 10% of the total number of rows from each of A and B
        mar_copy = data_to_mask.copy()
        mar_copy = mar_copy.mask(missing_indices_matrix == 1)

        return(mar_copy, missing_indices_matrix, summary_df)  
        
    def get_patterned_missing_indices_matrix(self, B_patterns, A_targets):
        """
        Function to get the missing indices matrix for the patterned overlaying given the sampled patterns and targets. 
        
        Args:
            B_patterns: list/1D array, indices of the rows with patterns to copy
            A_targets: list/ 1D array, indices of the rows where pattern is to be applied
            
        Returns:
            missing_indices_matrix: 2D numpy array, binary, with the indices of the masked entries
        """

        # create a matrix of 0s the same size as the dataframe
        missing_indices_matrix = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(len(B_patterns)):
            row_pattern = np.where(self.data.iloc[B_patterns[i]].isna(),
                                   1,
                                   0)
            missing_indices_matrix[A_targets[i], :] = row_pattern

        return(missing_indices_matrix)

    def hard_impute(self, O, Ω, r=1, tolerance=1e-3, max_iter=3000):
        """
        Function to perform the matrix completion. 
        
        Args:
            O: 2D numpy array, missing values replaced by 0's and non-na's are original values
            Ω: 2D numpy arry, binary matrix with 0 where the matrix has missing values and 1 otherwise
            r: int, number of factors used in teh imputation
            tolerance: float, tolerance value for managing convergence
            max_iter: int, maximum number of iterations allowed
            
        Returns:
            M: 2D numpy array of the completed matrix in the original dimensions
        """
        M = np.zeros_like(O)
        for T in range(max_iter):
            M_new = self.SVD(O * Ω + (1-Ω) * M , r)
            print('Iteration:', T, end = "\r")
            if (np.linalg.norm(M-M_new) < np.linalg.norm(M)*tolerance):
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
        u,s,vh = np.linalg.svd(M, full_matrices=False)
        s[r:] = 0
        M_fact = None
        return (u*s).dot(vh)
    
    def get_raw_omega_mats(self, use_full_matrix=False):
        """
        Helper function to generate the O and Ω matrices needed for the hard-impute function
        
        Args:
            masked_copy: pd.DataFrame, dataframe with values already masked
            
        Returns:
            omega_st: 2D numpy array needed for hard impute function as Ω
            raw_st: 2D numpy array needed for the hard impute function as O
        """
        omega_st = self.masked_copy.copy()
        raw_st = self.masked_copy.copy()
        
        if use_full_matrix: 
            omega_st = self.data.copy()
            raw_st = self.data.copy()
        # the .where function is counter-intuitive for dfs. Where the condition is NOT satisfied, replace the value with ... (default NA)
        omega_st = omega_st.where(~omega_st.isna(), 0)
        omega_st = omega_st.where(omega_st == 0, 1)
        
        
        raw_st = raw_st.where(~raw_st.isna(), 0)
        
        omega_st = np.array(omega_st)
        raw_st = np.array(raw_st)

        return omega_st, raw_st

    
    def r2_by_column(self, missing_indices_matrix, mar_matrix, true_df, imputed_matrix):
        """
        Function to find the performance of the imputation in terms of the r^2. Gives a boosting r2 and the raw r2

        Args:
            missing_indices_matrix: 2D array/dataframe, binary matrix of missing entries
            mar_matrix: pd.DataFrame, masked dataframe
            true_df: pd.DataFrame, non-masked dataframe
            imputed_matrix: 2D array, imputed matrix from matrix completion

        Outputs:
            r2_tot: float in (0, 1), overall r2 value for the imputation
            r2_columns: pd.DataFrame, columns for feature, imputation R2 and missing percentage
        """
        
        """
        Nov.17
        
        modify the function to calculate features with values of value > 0.01
        
        """
        '''
        features_lst = ['a_prior_freq_AvE_lag1',
                        'a_prior_freq_AvE_lag2',
                        'a_prior_freq_AvE_lag3',
                        'a_prior_freq_AvE_lag5',
                        'a_prior_loss_AvE_c_dc_lag1',
                        'a_prior_loss_AvE_c_dc_lag2',
                        'a_prior_loss_AvE_c_dc_lag3',
                        'a_prior_loss_AvE_c_dc_lag5',
                        'a_prior_loss_AvE_lag1',
                        'a_prior_loss_AvE_lag2',
                        'a_prior_loss_AvE_lag3',
                        'a_prior_loss_AvE_lag5',
                        'a_wage_ave_c']
        mar_matrix_features = 
        '''
        
        na_percs = []
        true_na_percs = []
        for column in mar_matrix.columns:
            
            na_percs.append(mar_matrix[column].isna().sum() / mar_matrix.shape[0])
            true_na_percs.append(true_df[column].isna().sum() / true_df.shape[0])

        r2_cols = []

        # initialise variables to keep track of the numerator and denominator in our overall R^2 calculation
        toptot = 0
        bottomtot = 0
        for i in range(missing_indices_matrix.shape[1]):
            
            #get validation x and y: indice = 0, masked
            valid_y = true_df.where(missing_indices_matrix != 0).iloc[:, i]
            valid_x = pd.DataFrame(imputed_matrix).where(missing_indices_matrix != 0).iloc[:, i]

            # now get the training x and y. Ensure we exclude the validation set, indice =1, not masked
            # we make sure we include only values that were not masked 
            train_y = true_df.where(missing_indices_matrix == 0).iloc[:, i]
            train_x = pd.DataFrame(imputed_matrix).where(missing_indices_matrix == 0).iloc[:, i]

            # set up easy to use dataframes for the regression
            train_df = pd.DataFrame({
                "X": train_x,
                "Y": train_y
            })
            train_df.dropna(inplace=True)

            valid_df = pd.DataFrame({
                "X": valid_x,
                "Y": valid_y
            })
            valid_df.dropna(inplace=True)
            
            # if we are in a column with no masked values 
            if valid_y.isna().sum() == true_df.shape[0]:
                r2 = 1
                top, bottom = 0, 0
                
            else:
                # train the regression on the training data
                model = LinearRegression()
                model.fit(np.array(train_df["X"]).reshape(-1, 1), np.array(train_df["Y"]))

                # get the OOS r^2 
                # make sure we use the mean from the training data in the denominator
                predictions = model.predict(np.array(valid_df["X"]).reshape(-1, 1))
                col_mean = np.mean(train_df["Y"]) 

                # use the r^2 formula
                top = np.nansum(np.subtract(predictions, valid_df["Y"]) ** 2)
                bottom = np.nansum((valid_df["Y"] - col_mean) ** 2) 
                r2 = 1 - top / bottom     

            r2_cols.append(r2)

            # keep track for our overall R^2
            toptot += top
            bottomtot += bottom

        # get the regular r2 as well as the boosting
        bottom = (true_df.where(missing_indices_matrix != 0).sub(true_df.mean(axis=0), axis=1) ** 2).sum(axis=0)
        top = ((pd.DataFrame(imputed_matrix, columns=self.data.columns).where(missing_indices_matrix != 0).sub(true_df.where(missing_indices_matrix != 0))) ** 2).sum(axis=0)

        r2_reals = 1 - top / bottom
        
        r2_by_column = pd.DataFrame({
            "Feature" : true_df.columns,
            "Missing_percentage" : na_percs,
            "Imputation_r2" : r2_cols,
            "Raw_r2_imputed":r2_reals
        }).sort_values(by=["Imputation_r2"]).reset_index(drop=True)
        
        r2_tot = r2 = 1 - toptot / bottomtot
        print('Overall R^2 value:')
        print(r2_tot)

        return(r2_tot, r2_by_column)
    
    def run_completion(self, num_factors, tolerance=1e-3, max_iter=3000, use_full_matrix=False):
        """
        Function to run the matrix completion for a given number of factors. Stores the r2 value and features, and 
        the imputed matrix as attributes of the MatrixCompletion object.

        Args:
            num_factors: int, the number of factors to be used in the matrix completion
            tolerance: float, tolerance for the hard impute function
            max_iter, int, maximum iterations for the hard impute function
            
        Outputs:
            Mat: 2D numpy array, imputed matrix
            r2: float in (0, 1), R^2 value for the imputation
            r2_cols, pd.DataFrame, dataframe of the imputation values for the matix completion including missingness
        """       
        
        print("Running matrix completion for {} factors:".format(num_factors))
        print()
        omega_st, raw_st = self.get_raw_omega_mats(use_full_matrix=False)
        Mat = self.hard_impute(raw_st, omega_st, r=num_factors, tolerance=tolerance, max_iter=max_iter)
        self.imputed_matrix = Mat
        self.num_factors = num_factors
        r2, r2_cols = self.r2_by_column(self.missing_indices_matrix,
                                        self.masked_copy,
                                        self.data,
                                        Mat)
        if use_full_matrix:
            print("Repeating completion without masking for {} factors".format(num_factors))
            omega_st, raw_st = self.get_raw_omega_mats(use_full_matrix=True)
            Mat = self.hard_impute(raw_st, omega_st, r=num_factors, tolerance=tolerance, max_iter=max_iter)
            self.imputed_matrix = Mat
        self.r2 = r2
        self.r2_cols = r2_cols
 
        return Mat, r2, r2_cols
    
    def sum_squares(self, u_i, x_i, V, Ω_i):
        """
        Helper function to provide the loss function to minimize for the online imputation. 
        Essentially the same as a regression, but we ignore the values of the missing entries to minimiize the sum of squares

        Args:
            u_i: 1D array, the new row of the U matrix
            x_i: 1D array, the new observation
            V: 2D array, recovered from the SVD
            Ω_i: binary 1D array, 0 where entries are missing
            
        Outputs:
            tot: float, the sum of squares
        """    
        tot = np.nansum(np.multiply(np.subtract(x_i.T, u_i.T @ V) ** 2, Ω_i))
        return tot
    
    def impute_new_row(self, new_observation, factors=None):
        """
        Function to perform online imputation given a new observation. Minimizes the loss function. 
        
        Args:
            factors: int, number of factors used in the imputation
            new_observation: 1D array, the new observation
            
        Outputs:
            imputed new observation
        """ 
        if factors is None:
            factors = self.num_factors

        # get the recovered U and V from the final matrix
        u, s, v = svds(self.imputed_matrix, k=factors)

        # get an initial value for u_i
        u_i0 = u.mean(axis=0)

        # make sure we have the right dimensions
        if new_observation.shape[0] != (u_i0.T @ v).shape[0]:
            new_observation = new_observation.T
        if new_observation.shape[0] != (u_i0.T @ v).shape[0]:
            print("New observation needs to be of size {} or {}, current input is size {}".format((u_i0.T @ v).shape, (u_i0.T @ v).T.shape, new_observation.T.shape))
            return None

        # set up the Ω row
        Ω_i = np.where(np.isnan(new_observation), 0, 1)

        # minimize
        u_i = optimize.minimize(self.sum_squares, x0=u_i0, args=(new_observation, v, Ω_i)).x

        imputed_row = u_i @ v
        return imputed_row
    
    def impute_new_data(self, new_data):
        """
        Function to perform online imputation given new data. Iteratively calls the function to impute a new row. 
        
        Args:
            new_data: pandas dataframe, the new observations. Can be one or many rows
            
        Outputs:
            completed new row (note: this replaces the actual observed values too and gives the reconstructed row) 
        """ 
        container = []
        for idx, row in new_data.iterrows():
            imputation = self.impute_new_row(np.array(row))
            container.append(imputation)
            
        new_df = pd.DataFrame(container, columns=new_data.columns)
        return new_df
        
        
    def run_completion_gridsearch(self, factor_range=range(2, 18), tolerance=1e-3, max_iter=3000):
        """
        Function to run the matrix completion for a given grid of factors. Stores the value of the best imputation
        and the the best r2 values, as well as the best matrix from the gridsearch. 

        Args:
            factor_range: array_like, factors for which to run the completion
            tolerance: float, tolerance for the hard impute function
            max_iter: int, maximum iterations for the hard impute function
            
        Outputs:
            Mat: 2D numpy array, imputed matrix
            r2_best: float in (0, 1), R^2 value for the imputation that performed best
            r2_cols, pd.DataFrame, dataframe of the imputation values for the matix completion including missingness
        """       
        r2s = []
        Ms = []
        r2_best = -100
        for i in factor_range:
            M, r2, r2_cols = self.run_completion(i, tolerance=tolerance)
            Ms.append(M)
            r2s.append(r2)
            if r2 > r2_best:
                r2_cols_best = r2_cols
        # choose the best number of factors and run again with even lower tolerance to ensure comlete convergence
        # having a lower tolerance takes longer, but imporves performance slightly for aggregate model
        max_r2 = max(r2s)
        best_index = r2s.index(max_r2)
        factors = factor_range[best_index]
        best_mat = Ms[best_index]

        print('Best number of factors:')
        print(factors)
        print("Best R^2")
        print(max_r2)
        
        self.gridsearch_r2s = r2s
        self.gridsearch_factors = factor_range
        self.gridsearch_best_mat = best_mat
        self.gridsearch_best_r2 = max_r2
        self.gridsearch_best_r2_cols = r2_cols_best
        
        self.imputed_matrix = best_mat
        self.num_factors = factors
        self.r2 = max_r2
        self.r2_cols = r2_cols_best
        return best_mat, r2_best, r2_cols_best

    def plot_gridsearch_results(self):
        """
        Function to plot the results of the gridsearch for matrix completion factors. 
        
        Assumes that the gridsearch has already been run.
        """
        # plot the line search results
        
        if self.gridsearch_r2s is None:
            print("First run completion gridsearch")
            return None

        fig, ax = plt.subplots(figsize = (15,9))
        mpl.rcParams['text.color'] = '#555555'

        x = self.gridsearch_factors
        r2s = self.gridsearch_r2s
        plt.plot(x, r2s, linewidth=3, color = '#1e9ab0')

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color('#4a4a4a')
        ax.spines["bottom"].set_color('#4a4a4a')

        ax.tick_params(color = '#555555', labelcolor = '#555555')
        plt.xticks(fontsize = 13, color = '#666666')
        plt.yticks(fontsize = 13, color = '#666666')
        plt.title('$R^{2}$ of Hard-Impute for Number of Factors', fontsize = 18, color = '#555555')
        ax.set_ylabel('\n$R^{2}$\n', rotation=90, position = (0.3,0.5), fontsize = 14, color = '#555555')
        ax.set_xlabel('\nFactors', fontsize = 14, color = '#555555')
        best_index = r2s.index(max(r2s))
        plt.plot(x[best_index], r2s[best_index],'o', color='red')
        plt.ylim([0, 1])
        # plt.text(x[best_index] - 0.5, r2s[best_index] + 0.2, '{} Factors'.format(x[best_index]), fontsize = 14, color = '#555555')
        plt.show()
        
    def plot_gridsearch_trends(self):
        """
        Function to plot R2 compared to missingness from the completion
        """
        
        # plot the trend of the r^2 values and the missingness of the data
        
        if self.gridsearch_r2s is None:
            print("First run completion gridsearch")
            return None

        r2_cols = self.gridsearch_best_r2_cols
        r2_cols.sort_values(by = ["Imputation_r2"], ascending=True, inplace=True)
        r2_cols.Imputation_r2 = np.where(r2_cols.Imputation_r2 < 0,
                                         0, 
                                         r2_cols.Imputation_r2)

        fig, ax = plt.subplots(figsize = (15,9))
        mpl.rcParams['text.color'] = '#555555'
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color('#4a4a4a')
        ax.spines["bottom"].set_color('#4a4a4a')
        
        x = self.gridsearch_factors
        r2s = self.gridsearch_r2s
        best_index = r2s.index(max(r2s))
    
        ax.tick_params(color = '#555555', labelcolor = '#555555')
        plt.xticks(fontsize = 13, color = '#666666')
        plt.yticks(fontsize = 13, color = '#666666')
        plt.title('$R^{}$ of features with {} factor aggregated model'.format(2, x[best_index]), fontsize = 18, color = '#555555')
        ax.set_ylabel('\n$R^{2}$\n', rotation=90, position = (0.3,0.5), fontsize = 14, color = '#555555')
        ax.set_xlabel('\nMissing Percentage of Feature', fontsize = 14, color = '#555555')
        plt.plot(r2_cols.Missing_percentage, r2_cols.Imputation_r2, 'o', color='#b01919', markersize=15)

        # plt.text(0.64, 0.582935, 'a_prior_loss_AvE_lag5', fontsize = 14, color = '#555555')
        # plt.text(0.15, 0.41, 'a_prior_loss_AvE_lag1', fontsize = 14, color = '#555555')
        plt.xlim([-0.05, 1])
        plt.ylim([-0.2, 1.2])
        plt.show()