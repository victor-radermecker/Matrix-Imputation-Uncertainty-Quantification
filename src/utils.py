import numpy as np
import multiprocessing as mp

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import multiprocessing as mp

# Pushing dataframe to S3 Bucket
import boto3
from io import StringIO
import time


from random import Random
from sklearn.linear_model import LinearRegression
from scipy.stats import t

import scipy
import scipy.stats as st
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 500)


# compute confidence interval
def student_cis(data, confidence_level):
    bounds = st.t.interval(
        confidence_level, len(data) - 1, loc=np.mean(data), scale=st.sem(data)
    )
    return bounds[0], np.mean(data), bounds[1]


def student_cis_centered(mean, std, center, nbr_points, confidence_level):
    bounds = st.t.interval(confidence_level, nbr_points - 1, loc=mean, scale=std**2)
    length = bounds[1] - bounds[0]
    lower_bound, upper_bound = center - length / 2, center + length / 2
    return lower_bound, center, upper_bound, length


def confidence_interval(data, center, confidence_level=0.95):
    n = len(data)
    # m = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_value = t.ppf((1 + confidence_level) / 2, n - 1)
    interval = t_value * std_err
    return center - interval, center, center + interval, 2 * interval


def scipy_cis_centered(mean, std, center, confidence_level):
    ci = scipy.stats.norm.interval(confidence_level, loc=mean, scale=std)
    print(ci)
    length = ci[1] - ci[0]
    return center - 0.5 * length, center, center + 0.5 * length, length


def partition(list_in, n):
    random.shuffle(list_in)
    res = np.array([list_in[i::n] for i in range(n)])
    return np.squeeze(res)


def a_b_split(matrix, a_prop):
    """
    Selects a fraction of non-missing indices in each column of the matrix and returns them.

    Args:
        matrix (numpy.ndarray): A matrix with shape (n_rows, n_cols).
        fraction (float): The fraction of non-missing indices to select in each column. Default is 0.1.

    Returns:
        A list of tuples, where each tuple contains the row and column indices of a selected non-missing value.
    """
    # Generate a mask of missing values
    missing_mask = np.isnan(matrix)

    # Generate a mask of non-missing values
    non_missing_mask = ~missing_mask

    # Generate a mask of fraction of non-missing values for each column
    a_mask = np.zeros_like(matrix, dtype=bool)
    for col in range(matrix.shape[1]):
        col_non_missing_indices = np.where(non_missing_mask[:, col])[0]
        num_val_indices = int(a_prop * len(col_non_missing_indices))
        validation_indices = np.random.choice(
            col_non_missing_indices, size=num_val_indices, replace=False
        )
        a_mask[validation_indices, col] = True

    # Get the indices of the validation set
    a_idx = list(zip(*np.where(a_mask)))

    # Get the indices of the training set
    b_mask = non_missing_mask & ~a_mask
    b_idx = list(zip(*np.where(b_mask)))

    return np.array(a_idx), np.array(b_idx)


def split_summary(matrix, val, train, name="val"):
    """
    Creates a pandas DataFrame with columns for the count of values in `val`, `train`,
    and missing values in `matrix`, and a total count of all values for each column.

    Args:
    - matrix (numpy.ndarray): A 2D array of values
    - val (list): A list of tuples containing the index and value of validation data
    - train (list): A list of tuples containing the index and value of training data

    Returns:
    - df (pandas.DataFrame): A DataFrame with columns 'val', 'train', 'miss', and 'total'
      containing the count of values in `val`, `train`, and missing values in `matrix`,
      and a total count of all values for each column.
    """
    val_counts = np.zeros(matrix.shape[1], dtype=np.int32)
    train_counts = np.zeros_like(val_counts)
    miss_counts = np.sum(np.isnan(matrix), axis=0)

    for idx, val_value in val:
        val_counts[val_value] += 1

    for idx, train_value in train:
        train_counts[train_value] += 1

    df = pd.DataFrame(
        {
            name: val_counts,
            "train": train_counts,
            "miss": miss_counts,
            "total": val_counts + train_counts + miss_counts,
        }
    )

    return df


def mask_matrix(matrix, idx):
    """
    Mask all elements in the input matrix except the ones at the specified indices, by setting their values to np.nan.

    Args:
        matrix (np.ndarray): Input 2D numpy array.
        idx (list): List of indices (as tuples) to be kept in the input matrix, all other elements will be masked.

    Returns:
        new_matrix (np.ndarray): A new 2D numpy array with the same shape as the input matrix,
                                 with all elements except the ones at the specified indices masked by setting their values to np.nan.
    """
    new_matrix = matrix.copy()
    mask = ~np.ones_like(matrix, dtype=bool)
    mask[tuple(zip(*idx))] = True
    new_matrix[~mask] = np.nan
    return new_matrix


def create_mask(matrix, idx):
    """
    This create_mask function takes a given matrix and a list of indices (train_idx) as input, and returns a boolean
    mask of the same shape as the input matrix. In the mask, the elements at the specified indices in train_idx are set
    to True, while all other elements are set to False.
    """
    idx = np.asarray(idx)
    mask = np.zeros(matrix.shape, dtype=bool)
    mask[tuple(idx.T)] = True
    return mask


def training_sample(matrix, train_idx, train_prop):
    masked_matrix = mask_matrix(matrix, train_idx)
    batch_idx, _ = a_b_split(masked_matrix, a_prop=train_prop)
    return batch_idx


def s3_save_csv(data, file_name="dataframe_", timestamp=False):

    if timestamp:
        # Keeping time for fileName
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = file_name + timestr

    # Creating S3 Resource From the Session.
    bucket_name = "cortex-mit1012-lmdl-workbucket"
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3 = boto3.resource("s3")
    s3.Object(bucket_name, f"mdata/{file_name}.csv").put(Body=csv_buffer.getvalue())


def s3_delete_file(file_name):

    bucket_name = "cortex-mit1012-lmdl-workbucket"
    s3 = boto3.resource("s3")
    print(f"Deleting mdata/{file_name}")
    s3.Object(bucket_name, f"mdata/{file_name}").delete()


def s3_load_csv(file_name):

    bucket_name = "cortex-mit1012-lmdl-workbucket"
    s3 = boto3.client("s3")
    object_key = f"mdata/{file_name}.csv"
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj["Body"]
    csv_string = body.read().decode("utf-8")
    df = pd.read_csv(StringIO(csv_string))
    return df


def s3_print_content():
    bucket_name = "cortex-mit1012-lmdl-workbucket"
    conn = boto3.client("s3")  # again assumes boto.cfg setup, assume AWS S3
    for key in conn.list_objects(Bucket=bucket_name)["Contents"]:
        print(key["Key"])


# @jit(nopython=True)
def hard_impute(O, Ω, r=1, tolerance=1e-3, max_iter=3000):
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
        M_new = SVD(O * Ω + (1 - Ω) * M, r)
        # print('Iteration:', T, end = "\r")
        if np.linalg.norm(M - M_new) < np.linalg.norm(M) * tolerance:
            break
        M = M_new
    return M


# @jit(nopython=True)
def SVD(M, r):
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
    M_fact = None
    return (u * s).dot(vh)


def extract_O(M):
    O = np.copy(M)
    O[np.isnan(O)] = 0
    return O


def extract_Omega(M):
    Omega = np.zeros(M.shape, dtype=int)
    Omega[~np.isnan(M)] = 1
    return Omega


def summarize_results(data, method):
    out = (
        data.groupby(["feature", "alpha"])["length", "cov", "std"].mean().reset_index()
    )
    out["method"] = method
    return out
