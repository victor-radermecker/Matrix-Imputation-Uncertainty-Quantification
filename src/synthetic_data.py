import numpy as np
import pandas as pd


class SyntheticDataset:
    """
    A class to generate synthetic datasets using low-rank matrix decomposition.
    """

    def __init__(
        self,
        n=1000,
        m=40,
        r=10,
        noise_std=0.1,
        min_prop=0.1,
        max_prop=0.3,
        num_categorical=0,
    ):
        """
        Initializes the synthetic dataset generator.

        Args:
        - n (int): The number of rows in the dataset
        - m (int): The number of columns in the dataset
        - r (int): The rank of the low-rank matrix decomposition
        - noise_std (float): The standard deviation of the noise added to the matrix
        - min_prop (float): The minimum proportion of missing values in each column
        - max_prop (float): The maximum proportion of missing values in each column
        """
        self.n = n
        self.m = m
        self.r = r
        self.noise_std = noise_std
        self.min_prop = min_prop
        self.max_prop = max_prop
        self.num_categorical = num_categorical

    def generate(self):
        """
        Generates a synthetic dataset with missing values.

        Returns:
        - synth (pd.DataFrame): A synthetic dataset with missing values as a Pandas DataFrame
        """
        U = np.random.rand(self.n, self.r)
        V = np.random.rand(self.m, self.r)
        M = U @ V.T

        E = np.random.normal(size=(self.n, self.m)) * self.noise_std
        O = M + E
        O = (O - np.mean(O)) / np.std(O)

        O = self._hide_values(O, self.min_prop, self.max_prop)

        synth = pd.DataFrame(O)

        # Convert the first `num_categorical` columns to categorical variables
        for i in range(self.num_categorical):
            synth[i] = pd.cut(
                synth[i], bins=np.linspace(-2, 2, 5), labels=["A", "B", "C", "D"]
            ).astype("category")

        col_names = [f"col{i}" for i in range(self.m)]
        synth.columns = col_names

        return synth

    def _hide_values(self, array, min_prop, max_prop):
        """
        Hides a random proportion of values in each column of a 2D numpy array by replacing
        them with NaN. The proportion of missing values in each column is randomly chosen
        between `min_prop` and `max_prop`.

        Args:
        - array (numpy.ndarray): A 2D array of values
        - min_prop (float): The minimum proportion of missing values in each column
        - max_prop (float): The maximum proportion of missing values in each column

        Returns:
        - masked_array (numpy.ndarray): A masked copy of the input array with missing values.
        """
        assert isinstance(array, np.ndarray), "Input array must be a numpy ndarray"
        assert array.ndim == 2, "Input array must be 2D"

        masked_array = array.copy()

        for i in range(array.shape[1]):
            prop = np.random.uniform(min_prop, max_prop)
            num_missing = int(prop * array.shape[0])
            indices = np.random.choice(array.shape[0], size=num_missing, replace=False)
            masked_array[indices, i] = np.nan

        return masked_array
