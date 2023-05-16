"""
.py script to prepare the data in the method specified by LM for their baseline model. 
Can be aggregated or not 

"""
import pandas as pd
import numpy as np
import h2o
h2o.init()
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from src import computegini as gn

# get the feature names used in the baseline model - from discussion between Tianyi and Jianying

first_feature_names = [
    'a_internal_wage_county',
    'a_internal_wage_st', 
    'a_onleveltrnd_lc_amt',
    "a_bls_wage_county",
    "a_bls_wage_st",
    'a_totpayroll8810_amt',
    "a_totpayroll_amt",
    'emod_calc',
    'a_dnb_score',
    "a_osha_insp_ct_10y",
    'a_OSHA_viol_ct_10y',
    'a_OSHA_Willful_Ct_10y',
    'a_OSHA_Repeat_Ct_10y',
    'a_prior_freq_AvE_lag1',
    'a_prior_freq_AvE_lag2',
    'a_prior_freq_AvE_lag3',
    'a_prior_freq_AvE_lag4',
    'a_prior_freq_AvE_lag5',
    'a_prior_freq_AvE_lag6',
    'a_prior_freq_AvE_lag7',
     'a_prior_loss_AvE_c_dc_lag1',
     'a_prior_loss_AvE_c_dc_lag2',
     'a_prior_loss_AvE_c_dc_lag3',
     'a_prior_loss_AvE_c_dc_lag4',
     'a_prior_loss_AvE_c_dc_lag5',
     'a_prior_loss_AvE_c_dc_lag6',
     'a_prior_loss_AvE_c_dc_lag7',
     'a_prior_loss_AvE_lag1',
     'a_prior_loss_AvE_lag2',
     'a_prior_loss_AvE_lag3',
     'a_prior_loss_AvE_lag4',
     'a_prior_loss_AvE_lag5',
     'a_prior_loss_AvE_lag6',
     'a_prior_loss_AvE_lag7',
     'loss_free_ct'
]

"""
data_preparation_baseline: 
Function to take in the unprocessed dataframe and output a processed one. 
Implements the LM data transformations specified and aggregates the data if needed. 
This frame will then be passed to the discretize_prepared_data function. 
Can be used for baseline or imputed model


Args:
- full_df: full feature df from LM at policy level, LM new data
- first_feature_names: names of the features for the baseline model (excluding factor variables)
- df_imputed: dataframe of imputed values 
- target_num: numerator of the target (not included in first_feature_names)
- target_den: denominator of the target (not included in first_feature_names)
- factor_vars
- aggregated: Boolean for whether we are aggregating at the account / term level

There is the possibility for flexibility with the target names and factor variables, but this hasn't been needed yet. Code might have to be adjusted if so. 

"""



def get_rolled_up_df(df, standardized=True):
    # data transformations done beforehand 
    df_agg = df.groupby(['Acct_Num', 'Pol_Eff_Date'], as_index=False).mean()

    # for aggregated, use df_agg.loc[:, first_feature_names], for non-aggregated use df.loc[:, first_feature_names]
    df_bucket_X = df_agg.loc[:, first_feature_names]

    # do the average wage transformation first because this was specified differently for dealing with missing info
    # a_wage_ave_c needs more special attention - this is a rough idea and the final product would be more precise / efficient or would perform a separate imputation before rolling up
    # we want to get a weighted average across all of the rows within accounts
    a_wage_ave_c = np.where((df.a_bls_wage_county.isna() | df.a_internal_wage_county.isna()),
                            df.a_internal_wage_st / df.a_bls_wage_st,
                            df.a_internal_wage_county / df.a_bls_wage_county)
    df_manip = df.copy()
    df_sum =  df_manip.groupby(['Acct_Num', 'Pol_Eff_Date'], as_index=False).sum()
    counts = df_manip.groupby(['Acct_Num', 'Pol_Eff_Date'], as_index=False).count().Pol_Num

    acct_sizes = []
    for idx, row in df_sum.iterrows():
        acct_sizes += [row.c_onleveltrnd_lc_amt] * counts[idx]

    df['weighted_wages'] = np.multiply(np.divide(df['c_onleveltrnd_lc_amt'], acct_sizes), a_wage_ave_c)
    df_sum =  df.groupby(['Acct_Num', 'Pol_Eff_Date'], as_index=False).sum()
    df_bucket_X['a_wage_ave_c'] = np.where(df_sum.weighted_wages == 0,
                                           np.nan,
                                           df_sum.weighted_wages)


    df_bucket_X.drop(['a_bls_wage_county', 'a_internal_wage_st', 'a_internal_wage_county', 'a_bls_wage_st'], axis=1, inplace=True)
    if not standardized:
        return df_bucket_X
    df_bucket_X = (df_bucket_X - df_bucket_X.mean()) / df_bucket_X.std()
    return df_bucket_X


def data_preparation_baseline(full_df, first_feature_names,
                              df_imputed=pd.DataFrame(), 
                              imputation_cols=[],
                              target_den='c_onleveltrnd_cap500k_lc_amt',
                              target_num='c_ultdevtrndloss_cap500k_amt',
                              factor_vars=['naics2_desc'],
                              aggregated=True
                            ):

    # Add the Naics description
    full_df['naics2_desc'] = full_df['naics2_desc'].fillna('na')
     
    
    full_df['c_ultdevtrndloss_cap500k_amt'] = np.where((full_df.c_ultdevtrndloss_cap500k_amt.isna()),
                                0,
                                full_df.c_ultdevtrndloss_cap500k_amt)
    
    # a_wage_ave_c: if we don't have the county wage, then use the state wage instead
    # as per Jianying's suggestion, when we roll up, we need to do a weighted average
    a_wage_ave_c = np.where((full_df.a_bls_wage_county.isna() | full_df.a_internal_wage_county.isna()),
                             full_df.a_internal_wage_st / full_df.a_bls_wage_st,
                             full_df.a_internal_wage_county / full_df.a_bls_wage_county)
    
    if not aggregated:  
        # select the relevant columns
        df_bucket_X = full_df.loc[:, first_feature_names]
        # standardise the data
        df_bucket_X = (df_bucket_X - df_bucket_X.mean()) / df_bucket_X.std()
        #bring in the target variables and the factor variables
        df_bucket_X['c_onleveltrnd_cap500k_lc_amt'] = full_df[target_den]
        df_bucket_X['c_ultdevtrndloss_cap500k_amt'] = full_df[target_num]
        df_bucket_X['naics2_desc'] = full_df[factor_vars]
        df_bucket_X['a_wage_ave_c'] = a_wage_ave_c
        df_bucket_X.drop(['a_bls_wage_county', 'a_internal_wage_st', 'a_internal_wage_county', 'a_bls_wage_st'],
                     axis=1, inplace=True)
    
    
    # aggregate data
    if aggregated:
        df_bucket_X = get_rolled_up_df(full_df)
        df_sum = full_df.groupby(['Acct_Num', 'Pol_Eff_Date', 'naics2_desc'], as_index=False).sum()
        df_bucket_X['c_onleveltrnd_cap500k_lc_amt'] = df_sum[target_den]
        df_bucket_X['c_ultdevtrndloss_cap500k_amt'] = df_sum[target_num]
        df_bucket_X['naics2_desc'] = df_sum[factor_vars]
        
    # do some feature engineering to match the LM baseline model   
    # input the imputed data, if there is any. Only add the imputation where we are missing the actual data
    for colname in imputation_cols:
        df_bucket_X[colname] = np.where((df_bucket_X[colname].isna()),
                                    df_imputed[colname],
                                    df_bucket_X[colname])
    
    # if we have no imputed data, then we naively use the mean imputation
    df_bucket_X.fillna(df_bucket_X.mean(), inplace=True)

    # some further feature transformations after we have imputed
    # first the target
    df_bucket_X['baseline_target'] = df_bucket_X.c_ultdevtrndloss_cap500k_amt / df_bucket_X.c_onleveltrnd_cap500k_lc_amt

    # a_payroll8810_pct_c
    df_bucket_X['a_payroll8810_pct_c'] = df_bucket_X.a_totpayroll8810_amt / df_bucket_X.a_totpayroll_amt

    # a_osha_viols_perinsp_10yrs_c
    df_bucket_X['a_osha_viols_perinsp_10yrs_c'] = df_bucket_X.a_OSHA_viol_ct_10y / df_bucket_X.a_osha_insp_ct_10y

    # a_osha_willful_perinsp_10yrs_c
    df_bucket_X['a_osha_willful_perinsp_10yrs_c'] = df_bucket_X.a_OSHA_Willful_Ct_10y / df_bucket_X.a_osha_insp_ct_10y

    # a_osha_willful_perinsp_10yrs_c
    df_bucket_X['a_osha_rptd_perinsp_10yrs_c'] = df_bucket_X.a_OSHA_Repeat_Ct_10y / df_bucket_X.a_osha_insp_ct_10y
    
    # drop the columns we no longer need
    df_bucket_X.drop(['a_totpayroll8810_amt', 'a_totpayroll_amt', 'a_OSHA_viol_ct_10y',
                      'a_OSHA_Willful_Ct_10y', 'a_OSHA_Repeat_Ct_10y', 'a_osha_insp_ct_10y'], axis=1, inplace=True)
        
    return(df_bucket_X)



"""
discretize_prepared_data: 
Function to take in the processed dataframe and output the training, validation and testing sets
implements the bucketing and one-hot encoding of variables and gives names to the variables. 
splits based on the partition variable provided by LM


Args:
- df_bucket_X: processed dataframe from the data_preparation_baseline function
- full_df: full feature df from LM at policy level
- num_buckets: the maximum discretized buckets for each feature
- target_vars: names of the variables to avoid when discretizing 
- factor_vars
- aggregated: Boolean for whether we are aggregating at the account / term level

There is the possibility for flexibility with the target names and factor variables, but this hasn't been needed yet. Code might have to be adjusted if so. 

"""



# function to discretize the data into buckets and return the data with bucket names

def discretize_prepared_data(df_bucket_X,
                             full_df,
                             num_buckets=5,
                             aggregated=True,
                             factor_vars=['naics2_desc'],
                             target_vars=['c_ultdevtrndloss_cap500k_amt', 'c_onleveltrnd_cap500k_lc_amt', 'baseline_target']
                            ):

    # suppress the warnings from having too few buckets:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    

    # set up the pipeline to make the dsicretized buckets
    transformer = make_pipeline(
        KBinsDiscretizer(n_bins=num_buckets, encode='onehot', strategy='quantile') # five buckets seems to be stable and produce good results 
    )
    
    # small trick to make naming the columns of the resultant dataframe easier
    discretizer = KBinsDiscretizer(n_bins=num_buckets, encode='onehot')

    #take out the columns not included 
    not_included = factor_vars + target_vars + ['Unnamed: 0']
    standardised = [e for e in df_bucket_X.columns.tolist()
                     if e not in not_included]

    # put the variables through the pipeline and the same for the discretizer to get the column names
    discretizer.fit(df_bucket_X.loc[:, standardised])
    column_trans_buckets = ColumnTransformer(
        [
            # ('standardising', StandardScaler(), standardised), 
            ("binned_numeric", transformer, standardised),
            (
                "onehot_categorical",
                OneHotEncoder(),
                factor_vars,
            ),
        ],
        remainder="drop",
    )

    bucket_X = column_trans_buckets.fit_transform(df_bucket_X)
    
    # stored most often as a sparse matrix and needs to be recovered
    try:
        bucket_X = (bucket_X.A)
    except:
        bucket_X = bucket_X  

    # recover the data and give it interpretable column names
    n_buckets = sum(discretizer.n_bins_)
    colnames = [f'{col}_{b}' for col, bins in zip(standardised, discretizer.n_bins_) for b in range(bins)] + ['naics2_desc_' + str(x) for x in range(bucket_X.shape[1] - n_buckets)]
    t = pd.DataFrame(bucket_X, columns = colnames)
    
    
    
    # add back the columns we let drop and the partition variable from the full dataframe
    t['c_ultdevtrndloss_cap500k_amt'] = df_bucket_X['c_ultdevtrndloss_cap500k_amt']
    t['c_onleveltrnd_cap500k_lc_amt'] = df_bucket_X['c_onleveltrnd_cap500k_lc_amt'] 
    t['baseline_target'] = df_bucket_X['baseline_target']
    
    # get the partition variables correct 
    if not aggregated:
        t['partition_var'] = full_df['a_partition_ind_2']
    if aggregated:
        grouped_df = full_df.groupby(['Acct_Num', 'Pol_Eff_Date', 'a_partition_ind_2'], as_index=False).mean()
        t['partition_var'] = grouped_df['a_partition_ind_2']

    building_df = t[t['partition_var'] == 'Building']
    testing_df = t[t['partition_var'] == 'Testing']
    holdout_df = t[t['partition_var'] == 'Holdout']
    
    return(building_df, testing_df, holdout_df)





"""
get_gini_scores: 
function to set up the dataframe for the gini_score
takes in the glm model and the data in the h2o dataset format
returns a triple of the gini, perfect gini and the normalized gini


Args:
- glm_model: trained GLM model
- valid: validation set as an h2o dataframe

"""

def get_gini_scores(glm_model, valid): 
    # make predictions and convert format 
    predictions = glm_model.predict(test_data=valid)
    pred_frame = predictions.as_data_frame()
    test_frame = valid.as_data_frame()
    
    # set up dataframes in the required format for teh gini calculations
    df1 = pd.DataFrame({'pred' : np.multiply(pred_frame["predict"], test_frame['c_onleveltrnd_cap500k_lc_amt']),
    'weight_var' : test_frame['c_onleveltrnd_cap500k_lc_amt'],
    'actual' : test_frame['c_ultdevtrndloss_cap500k_amt']
    })
    
    df2 = pd.DataFrame({'pred' : test_frame['c_ultdevtrndloss_cap500k_amt'],
    'weight_var' : test_frame['c_onleveltrnd_cap500k_lc_amt'],
    'actual' : test_frame['c_ultdevtrndloss_cap500k_amt']
    })
    
    gs1 = gn.GiniScore(data = df1, pred = "pred", actual = "actual", weight_var = 'weight_var', partition_var = None).giniscore().iloc[0, 1]
    gs2 = gn.GiniScore(data = df2, pred = "pred", actual = "actual", weight_var = 'weight_var', partition_var = None).giniscore().iloc[0, 1]
    gini_norm = gs1/gs2
    
    
    return(gs1, gs2, gini_norm)


# do some cross validation on the alpha value and lambda based on the gini 


"""
gini_grid_search: 
function to perform a grid search of the GLMs for the gini coefficient
Very simple function that runs the GLM for different combinations of alpha and lambda to find the highest gini


Args:
- alphas: array-like of alphas to search
- lambdas: array-like of lambdas to search
- train: training h2o dataframe
- valid: validation set h2o dataframe
- predict_cols: list of columns used to predict in the GLM model
- tweedie_power and tweedie_link: parameters for GLM set to LM defaults

"""

def gini_grid_search(alphas, lambdas, train, valid, predict_cols, tweedie_power=1.5, tweedie_link=0):
    h2o.no_progress()
    
    test_frame = valid.as_data_frame()
    ginis = []
    num_iterations = len(alphas) * len(lambdas)
    
    # iteration count to keep track of the progress of the gridsearch
    iter_count = 1
    for power in lambdas:
        
        current_row = []
        for alpha in alphas:
            print("Iteration ", iter_count, " of ", num_iterations, end = "\r")
            
            # run the model for each combination of alpha and lambda in the grid
            glm_model_local = H2OGeneralizedLinearEstimator(family= "tweedie",
                                              weights_column="c_onleveltrnd_cap500k_lc_amt",
                                              tweedie_link_power=tweedie_link,
                                              tweedie_variance_power=tweedie_power,
                                              alpha = alpha,
                                              lambda_ = power)
                                             # lambda_search=True)

            glm_model_local.train(predict_cols, "baseline_target", training_frame=train, validation_frame=valid)
            gs1, gs2, gini_norm = get_gini_scores(glm_model_local, valid)
            iter_count += 1

            current_row.append(gs1)
        ginis.append(current_row)

    # present the results in a dataframe and find the maximum gini, and the parameter values that achieved it
    results = pd.DataFrame(ginis).T
    results.columns = lambdas
    results["alphas"] = alphas
    results.set_index('alphas', inplace=True)
    max_lambda = results.max(axis=0).tolist().index(results.max().max())
    max_alpha = results.max(axis=1).tolist().index(results.max().max())
    print("Max gini: ", results.max().max(), "at λ = ", lambdas[max_lambda], ", α = ", alphas[max_alpha])
    
    return(results, lambdas[max_lambda], alphas[max_alpha], results.max().max())

