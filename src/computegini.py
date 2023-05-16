"""Calculates Gini score."""

import sys

import numpy as np
import pandas as pd


class GiniScore:
    """Calculate Gini score based on actual and predicted model target.

    Args:
        data: Dataframe containing predicted and actual target values
        pred: Predicted value column name (predicted loss amount)
        actual: Actual value column name (actual loss amount)
        weight_var: Weight variable column name (exposure amount)
        partition_var: Data partition variable column name

    Examples:
        Calculate gini score

        >>> import pandas as pd
        >>> import numpy as np
        >>> from import lmmod.evaluation.giniscore import GiniScore
        >>> df = pd.read_csv("./lmmod/data/UK_GL.csv")
        >>> gs = GiniScore(data = df, pred = "PredLossAmt", actual = "LossAmt", partition_var = "NewRenewalCode")
        >>> score = gs.giniscore()
    """

    def __init__(self, data, pred, actual, weight_var=None, partition_var=None):

        self.data = data.copy()
        self.pred = pred
        self.actual = actual
        self.weight_var = weight_var
        self.partition_var = partition_var

    def giniscore(self):
        """Generates Gini score."""
        if self.partition_var == None:
            self.partition_var = "splt"
            self.data[self.partition_var] = 1

        if self.weight_var == None:
            self.weight_var = "weight_var"
            self.data[self.weight_var] = 1

        if (
            self.data.loc[:, [self.pred, self.actual]]
            .agg(lambda x: x.isnull().any())
            .any()
        ):
            sys.exit("pred and actual columns can not have missing")

        if self.data.loc[:, [self.weight_var]].agg(lambda x: x.isnull().any()).any():
            sys.exit("weight_var (if specified) column can not have missing")

        lst = [
            self.data.loc[self.data[self.partition_var] == i]
            for i in self.data[self.partition_var].unique()
        ]
        partition_list = [i for i in self.data[self.partition_var].unique()]
        lst_len = len(lst)
        gini_w = list()

        for i in range(lst_len):
            data_split = lst[i].copy()

            # resolve ties in pred/weight
            data_split["r"] = round(
                data_split[self.pred] / data_split[self.weight_var], 8
            )
            data_split = data_split.groupby("r").sum().reset_index()
            # proceed with original code
            temp = (
                data_split.assign(r=lambda df: df[self.pred] / df[self.weight_var])
                .sort_values(by="r")
                .assign(
                    cum_w=lambda df: df[self.weight_var].cumsum()
                    / df[self.weight_var].sum()
                )
                .assign(
                    cum_l=lambda df: df[self.actual].cumsum() / df[self.actual].sum()
                )
                .pipe(lambda df: df.assign(gini=df["cum_w"] - df["cum_l"]))
            )

            gini_w.append(2 * np.average(temp["gini"], weights=temp[self.weight_var]))

        results = pd.DataFrame({"Group": partition_list, "Gini": gini_w})
        return results