# MIT Research: Confidence Intervals Packages

This Jupyter Notebook serves as a tutorial on how to use the matrix completion package developed by MIT for Liberty Mutual. The notebook includes the following features:

`Author:` Victor Radermecker

`Professor:` Prof. Vivek Farias

## Synthetic Dataset Generator

The `SyntheticDataset` class generates synthetic datasets using low-rank matrix decomposition and introduces missing values in the data. It is initialized with the following parameters:

- `n` (int): The number of rows in the dataset.
- `m` (int): The number of columns in the dataset.
- `r` (int): The rank of the low-rank matrix decomposition.
- `noise_std` (float): The standard deviation of the noise added to the matrix.
- `min_prop` (float): The minimum proportion of missing values in each column.
- `max_prop` (float): The maximum proportion of missing values in each column.
- `num_categorical` (int): The number of categorical columns in the dataset.

The `generate` method generates a synthetic dataset with missing values using low-rank matrix decomposition and returns it as a Pandas DataFrame. The first `num_categorical` columns are converted into categorical columns.

## Matrix Completion and Confidence Intervals

The `MatrixCompletionFast` class is designed for matrix completion and imputing missing values in a Pandas DataFrame. It is initialized with the following parameters:

- `data` (Pandas DataFrame): The input dataset to be imputed.
- `train_prop` (float): The proportion of the data to be used for training.
- `factors` (int): The number of factors to use for imputation.
- `nbr_batches` (int): The number of batches to divide the dataset into for parallel processing.
- `val_prop` (float): The proportion of the training data to be used for validation.
- `n_jobs`: The number of parallel jobs to use during matrix completion. Default is -1 (all available processors). (Note: This functionality is not fully implemented yet.)
- `random_seed`: The random seed to use for reproducibility.

### Matrix Imputation - Usage

To impute missing values in a Pandas DataFrame, follow these steps:

1. Import the `MatrixCompletion` class from the MCF package.
2. Create an instance of the `MatrixCompletion` class.
3. Use the `impute_dataframe()` function to impute missing values in the DataFrame. Set `keep_missing_only=True` if you want to impute only missing values.

Example:

```python
from MCF import MatrixCompletion
mc = MatrixCompletion()
imputed_df = mc.impute_dataframe(keep_missing_only=True)
```

### Confidence Intervals - Usage

To compute confidence intervals, initialize the confidence interval model with the following parameters:

- `conf_level`: The confidence level to use for the confidence intervals. It is recommended to provide a list of various confidence levels to choose the optimal one later.
- `beta`: The beta parameter allows artificially inflating the confidence intervals to account for uncertainties. A value of `beta=4` is suggested based on experiments.

Then, use the `confidence_intervals` method to compute the confidence intervals for the available methods: baseline (mean imputation), bootstrapping, and MIT uncertainty quantification paper.

Example:

```python
MCF.confidence_intervals(conf_lvls, beta=4)
```

The confidence interval computation may take some time, and upon successful completion, the message "Success. Exit." will be displayed. The results can be extracted using:

```python
baseline, bootstrap, mit = MCF.summarize_results()
```

Visualizations of the results can be obtained using:

```python
MCF.plot_results_feature()
```

*Note: This README provides an overview of the notebook and its functionalities. For detailed usage and examples, refer to the Jupyter Notebook itself.*
