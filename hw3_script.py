import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#data paths
base_dir = "/Users/antoninberanger/Documents/ECON473/homeworks/HW3"
data_dir = os.path.join(base_dir, "data")
plot_dir = os.path.join(base_dir, "graphs")

# helpers for next questions 
def weighted_variance(series, weights):
    if len(series) == 0: return np.nan
    average = np.average(series, weights=weights)
    variance = np.average((series - average)**2, weights=weights)
    return variance 

def weighted_mean(series, weights):
    if len(series) == 0: return np.nan
    return np.average(series, weights=weights)

# question 1
print("Loading data from HW3 directory...")
datasets = {
    1982: pd.read_csv(os.path.join(data_dir, 'scf-13M00004-E-1982-ind_F1.csv')),
    1990: pd.read_csv(os.path.join(data_dir, 'scf-13M00004-E-1990-ind_F1.csv')),
    2016: pd.read_csv(os.path.join(data_dir, 'CIS-72M0003-E-2016_F1.csv')),
    2018: pd.read_csv(os.path.join(data_dir, 'CIS-72M0003-E-2018_F1.csv'))
}

for year in [2016, 2018]:
    df = datasets[year]
    df['USHRWK'] = np.where(df['USHRWK'] >= 999.6, np.nan, df['USHRWK']) 
    df['WKSEM'] = np.where(df['WKSEM'] >= 96, np.nan, df['WKSEM'])
    for col in ['WGSAL', 'ATINC', 'TTINC']:
        if col in df.columns:
            df[col] = np.where(df[col] >= 99999996, np.nan, df[col])
    datasets[year] = df

for year, df in datasets.items():
    if year in [1982, 1990]:
        wage_var, hours_var, weeks_var, weight_var = 'WAGSAL', 'HRSWRK', 'WKSWRK', 'WEIGHT'
    else:
        wage_var, hours_var, weeks_var, weight_var = 'WGSAL', 'USHRWK', 'WKSEM', 'FWEIGHT'

    if weight_var in df.columns:
        df['final_weight'] = df[weight_var].fillna(0)
        if df['final_weight'].sum() == 0:
            df['final_weight'] = 1
    else:
        df['final_weight'] = 1
    
    df['hourly_wage'] = df[wage_var] / (df[hours_var] * df[weeks_var])
    datasets[year] = df

# question 2
cpi = {1982: 40.07, 1990: 57.23, 2016: 93.72, 2018: 97.37}
cpi_df = pd.DataFrame(list(cpi.items()), columns=['year', 'cpi'])

plt.figure(figsize=(8, 5))
plt.plot(cpi_df['year'], cpi_df['cpi'], marker='o', color='red')
plt.title('Evolution of CPI')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'cpi_evolution.png'))
plt.show()

# hourly real wage in 2020 dollars, used for all next questions
for year, df in datasets.items():
    df['real_wage'] = (df['hourly_wage'] / cpi[year]) * 100


# question 3
min_wages = {1982: 3.74, 1990: 5.05, 2016: 11.30, 2018: 13.41}

for year in [1982, 1990, 2016, 2018]:
    df = datasets[year]
    n_start = len(df)
    
    df = df.dropna(subset=['hourly_wage', 'final_weight'])
    
    limit = min_wages[year] * 0.5
    df = df[df['hourly_wage'] >= limit]
    n_after_wage = len(df)
    
    if year <= 1990:
        df = df[(df['AGE'] >= 25) & (df['AGE'] <= 60)]
        h_total = df['HRSWRK'] * df['WKSWRK']
    else:
        df = df[(df['AGEGP'] >= 7) & (df['AGEGP'] <= 14)]
        h_total = df['USHRWK'] * df['WKSEM']
        
    df = df[h_total > 260]
    n_final = len(df)
    
    print(f"\nSummary for {year}:")
    print(f"Initial: {n_start} | After Wage: {n_after_wage} | Final: {n_final}")
    
    datasets[year] = df

#question 4 & 5
for year, df in datasets.items():
    print(f"Generating Education and Experience for {year}...")

    if year == 1982:
        edu_map = {1: 10, 2: 10, 3: 10, 4: 12, 5: 12, 6: 12, 7: 13, 8: 16}
        df['yrs_schooling'] = df['EDUC'].map(edu_map)
        df['pot_exp'] = df['AGE'] - np.maximum(10, df['yrs_schooling']) - 6

    elif year == 1990:
        edu_map = {1: 10, 2: 10, 3: 10, 4: 12, 5: 12, 6: 13, 7: 16}
        df['yrs_schooling'] = df['RECEDUC'].map(edu_map)
        df['pot_exp'] = df['AGE'] - np.maximum(10, df['yrs_schooling']) - 6

    else: 
        edu_map = {1: 10, 2: 12, 3: 13, 4: 16}
        df['yrs_schooling'] = df['HLEV2G'].map(edu_map)
        age_group_to_midpoint = {7: 27, 8: 32, 9: 37, 10: 42, 11: 47, 12: 52, 13: 57, 14: 62}
        df['AGE_numeric'] = df['AGEGP'].map(age_group_to_midpoint)
        df['pot_exp'] = df['AGE_numeric'] - np.maximum(10, df['yrs_schooling']) - 6

    df['yrs_schooling'] = df['yrs_schooling'].fillna(10)
    df['pot_exp'] = df['pot_exp'].clip(lower=0)
    df = df.dropna(subset=['real_wage', 'yrs_schooling', 'pot_exp'])
    datasets[year] = df

# question 6
summary_stats = []

for year in [1982, 1990, 2016, 2018]:
    df = datasets[year]
    
    w = df['final_weight']
    # creating the log wage for all next questions
    df['log_real_wage'] = np.log(df['real_wage'])
    
    stats = {
        'Year': year,
        'Mean Real Wage': weighted_mean(df['real_wage'], w),
        'Log Var Wages': weighted_variance(df['log_real_wage'], w),
        'Mean Schooling': weighted_mean(df['yrs_schooling'], w),
        'Var Schooling': weighted_variance(df['yrs_schooling'], w),
        'Mean Experience': weighted_mean(df['pot_exp'], w),
        'Var Experience': weighted_variance(df['pot_exp'], w)
    }
    
    summary_stats.append(stats)

final_table = pd.DataFrame(summary_stats)

pd.options.display.float_format = '{:,.4f}'.format
print("\nQuestion 6 Table")
print(final_table.to_string(index=False))

output_path = os.path.join(base_dir, 'summary_table_q6.csv')
final_table.to_csv(output_path, index=False)


# miner regression helper function for all next questions 
def run_mincer_regression(df):
    """
    Fits a Mincerian WLS regression and returns the results object.
    Expects df to have: log_real_wage, yrs_schooling, pot_exp, final_weight
    """
    df = df.copy()
    df['pot_exp_sq'] = df['pot_exp'] ** 2
    
    Y = df['log_real_wage']
    X = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq']])
    w = df['final_weight']
    
    model = sm.WLS(Y, X, weights=w, missing='drop')
    results = model.fit()
    
    return results

# question 7
regression_results = {}

for year in [1982, 1990, 2016, 2018]:
    # Use the helper
    results = run_mincer_regression(datasets[year])
    
    # Store the statistics for your table
    regression_results[year] = {
        'Constant': results.params['const'],
        'Years of Schooling': results.params['yrs_schooling'],
        'Potential Exp': results.params['pot_exp'],
        'Potential Exp^2': results.params['pot_exp_sq'],
        'R-squared': results.rsquared,
        'Observations': int(results.nobs)
    }

print("\nQuestion 7a Regression Results")
for year, res in regression_results.items():
    print(f"\nYear: {year}")
    for key, value in res.items():
        print(f"{key}: {value:.4f}")

#B 
results = regression_results[2018]
mean_experience = final_table.loc[final_table['Year'] == 2018, 'Mean Experience'].values[0]
marginal_return = results['Potential Exp'] + 2 * results['Potential Exp^2'] * mean_experience

print(f"\nQuestion 7b: Marginal return to experience in 2018 at mean experience level ({mean_experience:.2f} years) is {marginal_return:.4f}")


#question 8 

#create new variables
for year, df in datasets.items():
    df['pot_exp_sq'] = df['pot_exp'] ** 2
    
    if 'SEX' in df.columns:
        df['female'] = np.where(df['SEX'] == 2, 1, 0)
    else:
        df['female'] = 0 
        
    df['edu_exp'] = df['yrs_schooling'] * df['pot_exp']
    
    datasets[year] = df.dropna(subset=['log_real_wage', 'yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female', 'edu_exp'])

# 1st regression : Base Mincer + Female Control
print("\nRegression 1: Base Mincer + Female Control\n")

spec1_results = {}

for year in [1982, 1990, 2016, 2018]:
    df = datasets[year]
    Y = df['log_real_wage']
    X1 = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female']])
    w = df['final_weight']
    
    model1 = sm.WLS(Y, X1, weights=w, missing='drop').fit()
    spec1_results[year] = {
        'Constant': model1.params['const'],
        'Years of Schooling': model1.params['yrs_schooling'],
        'Potential Exp': model1.params['pot_exp'],
        'Potential Exp^2': model1.params['pot_exp_sq'],
        'Female Dummy': model1.params['female'],
        'R-squared': model1.rsquared,
        'Observations': int(model1.nobs)
    }

spec1_table = pd.DataFrame(spec1_results)
pd.options.display.float_format = '{:,.4f}'.format
print(spec1_table.to_string())

# 2nd regression : Base Mincer + Female Control + Interaction Term
print("\nRegression 2: Base Mincer + Female Control + (Education x Experience)\n")

spec2_results = {}

for year in [1982, 1990, 2016, 2018]:
    df = datasets[year]
    Y = df['log_real_wage']
    X2 = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female', 'edu_exp']])
    w = df['final_weight']
    
    model2 = sm.WLS(Y, X2, weights=w, missing='drop').fit()
    spec2_results[year] = {
        'Constant': model2.params['const'],
        'Years of Schooling': model2.params['yrs_schooling'],
        'Potential Exp': model2.params['pot_exp'],
        'Potential Exp^2': model2.params['pot_exp_sq'],
        'Female Dummy': model2.params['female'],
        'Edu x Exp': model2.params['edu_exp'],
        'R-squared': model2.rsquared,
        'Observations': int(model2.nobs)
    }

spec2_table = pd.DataFrame(spec2_results)
print(spec2_table.to_string())

#########
# answer question 8 
"""
Briefly comment on why you chose your specifications, which specification you prefer, and why.
"""
#########

#question 9
#mincer regression
"""
log(wage) = β_0 + β_1 * schooling + β_2 * experience + β_3 * experience^2 + epsilon

In the Mincer regression, the log of wages is modeled as a function of schooling and experience (both linear and squared), 
with an error term (epsilon) capturing unobserved factors.

1. Changes in Attributes: If the average level of schooling or experience in the population changes, 
this will affect the average log wage. For example, if more people attain higher education, 
the average log wage would increase due to a higher β_1 * schooling term.

2. Changes in Returns: If the coefficients (β_1, β_2, β_3) change over time, 
this indicates that the labor market is valuing these attributes differently. 
For instance, if the return to schooling (β_1) increases, even if the average schooling level remains the same, 
the average log wage would increase because each year of schooling is now worth more in terms of wages.

3. Changes in Residuals: The error term captures unobserved factors that affect wages. 
If the distribution of these unobserved factors changes over time, this could also lead to changes in the average log wage, 
independent of changes in attributes or returns.

"""

#question 10

# get R-squared values for the base Mincer regression
print("\nR-squared for Base Mincer Regression:\n")
for year in [1982, 1990, 2016, 2018]:
    r2 = regression_results[year]['R-squared']
    print(f"{year}: {r2:.4f}")

years = [1982, 1990, 2016, 2018]
explained_vars = []
unexplained_vars = []

print("\nExplanatory power of the regression:\n")
# total variance of log wages can be decomposed into the part explained by the regression (R-squared) and the unexplained part (1 - R-squared).

for year in years:
    r_squared = regression_results[year]['R-squared']
    
    total_log_var = final_table.loc[final_table['Year'] == year, 'Log Var Wages'].values[0]
    
    exp_var = total_log_var * r_squared
    unexp_var = total_log_var * (1 - r_squared)
    
    explained_vars.append(exp_var)
    unexplained_vars.append(unexp_var)
    
    print(f"{year}: Total Var = {total_log_var:.4f} | Explained = {exp_var:.4f} | Unexplained = {unexp_var:.4f}")

plt.figure(figsize=(10, 6))

plt.plot(years, explained_vars, marker='o', color='blue', linewidth=2, label='Explained Variance')

plt.plot(years, unexplained_vars, marker='s', color='red', linewidth=2, label='Unexplained Variance')

plt.title('Explained vs Unexplained Variance of Log Wages', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Variance', fontsize=12)
plt.xticks(years)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_plot = os.path.join(plot_dir, 'variance_decomposition.png')
plt.savefig(output_plot, dpi=300)
plt.show()

"""
The unexplained variance increased much more than the explained variance. 
Between 1982 and 2018, the explained inequality grew by roughly 0.0096 (from 0.0244 to 0.0340), 
while the unexplained inequality grew by 0.0556 (from 0.2602 to 0.3158). 
This means that while education and experience play a role, the vast majority of the rise in Canadian wage inequality 
over this period was driven by unobserved factors (technological change, decline in unionization, 
or shifts in industry premiums) that are not captured in the basic Mincer equation.
"""

#question 11

#add a 2018 dummy variable to the 1982 data
df_1982 = datasets[1982].copy()
df_1982['year_2018'] = 0

df_2018 = datasets[2018].copy()
df_2018['year_2018'] = 1

combined_df = pd.concat([df_1982, df_2018], ignore_index=True)

combined_df['school_x_2018'] = combined_df['yrs_schooling'] * combined_df['year_2018']

Y = combined_df['log_real_wage']
X = sm.add_constant(combined_df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'year_2018', 'school_x_2018']])
w = combined_df['final_weight']

model = sm.WLS(Y, X, weights=w, missing='drop').fit()
print("\nQuestion 11 Regression Results:\n")
print(model.summary())

"""
Interaction term (years of schooling x 2018) => get the p-value for the change in return to schooling between 1982 and 2018
seems to be the correct regression to run but the interaction term is not significant (P>|t| = 0.49). 
This suggests that the return to schooling did not significantly change between 1982 and 2018 ?
"""

#question 12
"""
I am choosing 2017 here since you are all usingthat survey? we are all using 2018 no?
"""
beta_1982 = [
    regression_results[1982]['Constant'],
    regression_results[1982]['Years of Schooling'],
    regression_results[1982]['Potential Exp'],
    regression_results[1982]['Potential Exp^2']
]

df_2018 = datasets[2018].copy()
df_2018['pot_exp_sq'] = df_2018['pot_exp'] ** 2

X_2018 = sm.add_constant(df_2018[['yrs_schooling', 'pot_exp', 'pot_exp_sq']])

predicted_log_wages_1982_prices = np.dot(X_2018, beta_1982)

explained_ineq_quest_12 = weighted_variance(
    predicted_log_wages_1982_prices, 
    df_2018['final_weight']
)

print("\nQuestion 12 Analysis:\n")
print(f"Explained Inequality in 1982 (Actual): {explained_vars[0]:.4f}")
print(f"Explained Inequality in 2018 (Actual): {explained_vars[3]:.4f}")
print(f"Explained Inequality (2018 attributes at 1982 prices): {explained_ineq_quest_12:.4f}")

"""
The explained inequality in 2018 using 1982 returns is 0.0240, which is very close to the actual explained inequality in 1982 (0.0244). 
This suggests that the changes in the distribution of attributes (schooling and experience) between 1982 and 2018 would have led to a 
similar level of explained wage inequality if the returns to those attributes had remained constant at their 1982 levels.
Therefore, the increase in explained wage inequality from 1982 to 2018 is largely driven by changes in the returns to schooling and experience, 
rather than changes in the distribution of those attributes. This reinforces the conclusion from question 10 that rising returns to human capital 
are a key factor in the evolution of wage inequality over this period.
"""

# question 13 14 
residuals_dict = {}
residuals_variance = {}

for year in [1982, 1990, 2016, 2018]:
    results = run_mincer_regression(datasets[year])
    
    year_residuals = results.resid
    residuals_dict[year] = year_residuals 
    
    weights_used = results.model.weights
    resid_var = weighted_variance(year_residuals, weights_used)
    
    total_var = final_table.loc[final_table['Year'] == year, 'Log Var Wages'].values[0]
    fraction = resid_var / total_var
    
    residuals_variance[year] = {
        'Residual Var': resid_var,
        'Fraction': fraction
    }

print("\nQuestion 13: Residuals from Mincer Regressions\n")
for year in [1982, 1990, 2016, 2018]:
    print(f"Sample residuals for {year}: \n{residuals_dict[year][:5]}")

print("\nQuestion 14: Variance of Residuals and Fraction of Total Inequality\n")
for year in [1982, 1990, 2016, 2018]:
    stats = residuals_variance[year]
    print(f"Year {year}:")
    print(f"  Unexplained Variance: {stats['Residual Var']:.4f}")
    print(f"  Fraction of Total Variance: {stats['Fraction']:.2%}")

"""
The variance of the residuals from the Mincer regression increased from 0.2602 in 1982 to 0.3158 in 2018, which is a significant increase. 
The fraction of total wage variance that is unexplained by the Mincer regression stayed constant over time, with a spike in 1990 that is not significant. 
This may suggest that in the 1990 sample, education and experience had almost no predictive power, or the data for that year was much noisier than the other years. 
This high percentage indicates that the vast majority of wage variation in Canada occurs between individuals who possess the same observed levels of education and experience. 
This statement seems valid for all year, with no clear trend over time. This reinforces the conclusion that while education and experience are important determinants of wages, 
there are many other unobserved factors (such as industry, occupation, skills, discrimination, or luck) that play a much larger role in driving wage inequality.
"""

# question 15

# question 16: 
model_A_results = {}
model_B_results = {}
model_C_results = {}

# Map the numeric codes to province abbreviations for readability
prov_map = {
    10: 'NL', 11: 'PE', 12: 'NS', 13: 'NB', 24: 'QC', 
    35: 'ON', 46: 'MB', 47: 'SK', 48: 'AB', 59: 'BC'
}

for year in [1982, 1990, 2016, 2018]:
    df = datasets[year].copy()
    
    # Base variables
    df['pot_exp_sq'] = df['pot_exp'] ** 2
    if 'SEX' in df.columns:
        df['female'] = np.where(df['SEX'] == 2, 1, 0)
    else:
        df['female'] = 0 
        
    # quadratic schooling
    df['yrs_schooling_sq'] = df['yrs_schooling'] ** 2
    
    # marital status dummy
    if year in [1982, 1990]:
        marst_col = 'MARSTAT'
    elif year == 2016:
        marst_col = 'MARST'
    else:
        marst_col = 'MARSTP'
        
    if marst_col in df.columns:
        df['married'] = np.where(df[marst_col].isin([1, 2]), 1, 0)
    else:
        print(f"Warning: Column {marst_col} not found in {year} data.")
        df['married'] = np.nan
        
    # province dummies
    if year == 1982:
        prov_col = 'GEOCODE'
    elif year == 1990:
        prov_col = 'PROVINCE'
    else:
        prov_col = 'PROV'
        
    if prov_col in df.columns:
        df['prov_name'] = df[prov_col].map(prov_map)
        prov_dummies = pd.get_dummies(df['prov_name'], prefix='prov').astype(int)
        if 'prov_ON' in prov_dummies.columns:
            prov_dummies = prov_dummies.drop('prov_ON', axis=1) # Ontario as reference
        
        df = pd.concat([df, prov_dummies], axis=1)
        prov_cols = prov_dummies.columns.tolist()
    else:
        prov_cols = []

    cols = ['log_real_wage', 'yrs_schooling', 'yrs_schooling_sq', 'pot_exp', 'pot_exp_sq', 
            'female', 'married', 'final_weight'] + prov_cols
    df = df.dropna(subset=cols)
    
    Y = df['log_real_wage']
    w = df['final_weight']
    
    # model A: Quadratic Schooling
    XA = sm.add_constant(df[['yrs_schooling', 'yrs_schooling_sq', 'pot_exp', 'pot_exp_sq', 'female']])
    model_A = sm.WLS(Y, XA, weights=w, missing='drop').fit()
    
    model_A_results[year] = {
        'Constant':              model_A.params['const'],
        'Schooling':             model_A.params['yrs_schooling'],
        'P-val (Schooling)':     model_A.pvalues['yrs_schooling'],
        'Schooling^2':           model_A.params['yrs_schooling_sq'],
        'P-val (Schooling^2)':   model_A.pvalues['yrs_schooling_sq'],
        'Exp':                   model_A.params['pot_exp'],
        'P-val (Exp)':           model_A.pvalues['pot_exp'],
        'Exp^2':                 model_A.params['pot_exp_sq'],
        'P-val (Exp^2)':         model_A.pvalues['pot_exp_sq'],
        'Female':                model_A.params['female'],
        'P-val (Female)':        model_A.pvalues['female'],
        'R-squared':             model_A.rsquared,
        'Observations':          int(model_A.nobs)
    }
    
    #model B: Marital Status
    XB = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female', 'married']])
    model_B = sm.WLS(Y, XB, weights=w, missing='drop').fit()
    
    model_B_results[year] = {
        'Constant':              model_B.params['const'],
        'Schooling':             model_B.params['yrs_schooling'],
        'P-val (Schooling)':     model_B.pvalues['yrs_schooling'],
        'Exp':                   model_B.params['pot_exp'],
        'P-val (Exp)':           model_B.pvalues['pot_exp'],
        'Exp^2':                 model_B.params['pot_exp_sq'],
        'P-val (Exp^2)':         model_B.pvalues['pot_exp_sq'],
        'Female':                model_B.params['female'],
        'P-val (Female)':        model_B.pvalues['female'],
        'Married':               model_B.params['married'],
        'P-val (Married)':       model_B.pvalues['married'],
        'R-squared':             model_B.rsquared,
        'Observations':          int(model_B.nobs)
    }
    
    # model C: Province Dummies
    XC = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female'] + prov_cols])
    model_C = sm.WLS(Y, XC, weights=w, missing='drop').fit()
    
    model_C_results[year] = {
        'Constant':              model_C.params['const'],
        'Schooling':             model_C.params['yrs_schooling'],
        'P-val (Schooling)':     model_C.pvalues['yrs_schooling'],
        'Exp':                   model_C.params['pot_exp'],
        'P-val (Exp)':           model_C.pvalues['pot_exp'],
        'Exp^2':                 model_C.params['pot_exp_sq'],
        'P-val (Exp^2)':         model_C.pvalues['pot_exp_sq'],
        'Female':                model_C.params['female'],
        'P-val (Female)':        model_C.pvalues['female'],
        'Alberta':               model_C.params.get('prov_AB', np.nan),
        'P-val (Alberta)':       model_C.pvalues.get('prov_AB', np.nan),
        'BC':                    model_C.params.get('prov_BC', np.nan),
        'P-val (BC)':            model_C.pvalues.get('prov_BC', np.nan),
        'Manitoba':              model_C.params.get('prov_MB', np.nan),
        'P-val (Manitoba)':      model_C.pvalues.get('prov_MB', np.nan),
        'New Brunswick':         model_C.params.get('prov_NB', np.nan),
        'P-val (New Brunswick)': model_C.pvalues.get('prov_NB', np.nan),
        'Newfoundland':          model_C.params.get('prov_NL', np.nan),
        'P-val (Newfoundland)':  model_C.pvalues.get('prov_NL', np.nan),
        'Nova Scotia':           model_C.params.get('prov_NS', np.nan),
        'P-val (Nova Scotia)':   model_C.pvalues.get('prov_NS', np.nan),
        'PEI':                   model_C.params.get('prov_PE', np.nan),
        'P-val (PEI)':           model_C.pvalues.get('prov_PE', np.nan),
        'Quebec':                model_C.params.get('prov_QC', np.nan),
        'P-val (Quebec)':        model_C.pvalues.get('prov_QC', np.nan),
        'Saskatchewan':          model_C.params.get('prov_SK', np.nan),
        'P-val (Saskatchewan)':  model_C.pvalues.get('prov_SK', np.nan),
        'R-squared':             model_C.rsquared,
        'Observations':          int(model_C.nobs)
    }
    
    # vif test for 2018
    if year == 2018:
        def calculate_vif(X_matrix):
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_matrix.columns
            vif_data["VIF"] = [variance_inflation_factor(X_matrix.values, i) for i in range(X_matrix.shape[1])]
            return vif_data
        
        vif_A = calculate_vif(XA)
        vif_B = calculate_vif(XB)
        vif_C = calculate_vif(XC)

print("\nRegression A: Quadratic Schooling\n")
print(pd.DataFrame(model_A_results).to_string(float_format='{:,.4f}'.format))

print("\nRegression B: Marital Status (Marriage Premium)\n")
print(pd.DataFrame(model_B_results).to_string(float_format='{:,.4f}'.format))

print("\nRegression C: Provinces (Reference: Ontario)\n")
print(pd.DataFrame(model_C_results).to_string(float_format='{:,.4f}'.format))

print("\n" + "="*50)
print("COLLINEARITY TEST (VIF) FOR 2018")
print("="*50)
print("\nModel A VIFs:")
print(vif_A.to_string(float_format='{:,.2f}'.format))
print("\nModel B VIFs:")
print(vif_B.to_string(float_format='{:,.2f}'.format))
print("\nModel C VIFs:")
# Filter out base variables just to show the provinces clearly
print(vif_C[vif_C['Variable'].str.contains('prov')].to_string(float_format='{:,.2f}'.format))

# Question 16 - Model D: Industry Dummies (1982 and 1990 only)
model_D_results = {}

ind_map = {
    1: 'Agriculture', 2: 'Other Primary', 3: 'Manuf_NonDur', 4: 'Manuf_Dur',
    5: 'Construction', 6: 'Transport_Comm', 7: 'Wholesale', 8: 'Retail',
    9: 'Finance', 10: 'Community_Svc', 11: 'Personal_Svc',
    12: 'Business_Svc', 13: 'Public_Admin'
    # 14 and 15 excluded (never worked / last worked 5+ years ago)
}

for year in [1982, 1990]:
    df = datasets[year].copy()
    df['pot_exp_sq'] = df['pot_exp'] ** 2
    df['female'] = np.where(df['SEX'] == 2, 1, 0)

    # Province dummies (Ontario as reference)
    if year == 1982:
        prov_col = 'GEOCODE'
    else:
        prov_col = 'PROVINCE'

    if prov_col in df.columns:
        df['prov_name'] = df[prov_col].map(prov_map)
        prov_dummies = pd.get_dummies(df['prov_name'], prefix='prov').astype(int)
        if 'prov_ON' in prov_dummies.columns:
            prov_dummies = prov_dummies.drop('prov_ON', axis=1)
        df = pd.concat([df, prov_dummies], axis=1)
        prov_cols = prov_dummies.columns.tolist()
    else:
        prov_cols = []

    # Industry dummies (Agriculture as reference)
    if year == 1982:
        ind_col = 'INDCODE'
    else:
        ind_col = 'IND'

    df['ind_name'] = df[ind_col].map(ind_map)
    df = df[df['ind_name'].notna()] 
    ind_dummies = pd.get_dummies(df['ind_name'], prefix='ind').astype(int)
    if 'ind_Agriculture' in ind_dummies.columns:
        ind_dummies = ind_dummies.drop('ind_Agriculture', axis=1)
    df = pd.concat([df, ind_dummies], axis=1)
    ind_cols = ind_dummies.columns.tolist()

    # Drop missing
    cols = ['log_real_wage', 'yrs_schooling', 'pot_exp', 'pot_exp_sq',
            'female', 'final_weight'] + prov_cols + ind_cols
    df = df.dropna(subset=cols)

    Y = df['log_real_wage']
    w = df['final_weight']

    XD = sm.add_constant(df[['yrs_schooling', 'pot_exp', 'pot_exp_sq', 'female']
                             + prov_cols + ind_cols])
    model_D = sm.WLS(Y, XD, weights=w, missing='drop').fit()

    model_D_results[year] = {
        'Constant':               model_D.params['const'],
        'Schooling':              model_D.params['yrs_schooling'],
        'P-val (Schooling)':      model_D.pvalues['yrs_schooling'],
        'Exp':                    model_D.params['pot_exp'],
        'P-val (Exp)':            model_D.pvalues['pot_exp'],
        'Exp^2':                  model_D.params['pot_exp_sq'],
        'P-val (Exp^2)':          model_D.pvalues['pot_exp_sq'],
        'Female':                 model_D.params['female'],
        'P-val (Female)':         model_D.pvalues['female'],
    }

    # Add all province and industry coefficients dynamically
    for col in prov_cols + ind_cols:
        model_D_results[year][col] = model_D.params.get(col, np.nan)
        model_D_results[year][f'P-val ({col})'] = model_D.pvalues.get(col, np.nan)

    model_D_results[year]['R-squared']   = model_D.rsquared
    model_D_results[year]['Observations'] = int(model_D.nobs)

print("\nRegression D: Female + Province + Industry (1982 & 1990)")
print("Note: *** p<0.01  ** p<0.05  * p<0.10  |  ref: Ontario, Agriculture")
print(f"{'Variable':<28} {'1982':>12} {'1990':>12}")
print("-" * 54)

print("--- Base Controls ---")
print(f"{'Schooling':<28} {'0.0698***':>12} {'0.0852***':>12}")
print(f"{'Female':<28} {'-0.2945***':>12} {'-0.2644***':>12}")

print("--- Selected Provinces (ref: Ontario) ---")
print(f"{'Alberta':<28} {'0.0635***':>12} {'-0.0809***':>12}")  
print(f"{'British Columbia':<28} {'0.1173***':>12} {'-0.0665***':>12}")  
print(f"{'Quebec':<28} {'0.0349***':>12} {'-0.0525***':>12}")  
print(f"{'Nova Scotia':<28} {'-0.1509***':>12} {'-0.1637***':>12}")  
print(f"{'PEI':<28} {'-0.2125***':>12} {'-0.1967***':>12}")  

print("--- Selected Industries (ref: Agriculture) ---")
print(f"{'Other Primary':<28} {'0.5904***':>12} {'0.5799***':>12}")  
print(f"{'Transport & Comm.':<28} {'0.5548***':>12} {'0.4566***':>12}")  
print(f"{'Manufacturing (Dur.)':<28} {'0.4969***':>12} {'0.3972***':>12}")  
print(f"{'Retail Trade':<28} {'0.2850***':>12} {'0.2095***':>12}")  
print(f"{'Personal Services':<28} {'0.4255***':>12} {'0.0152':>12}")  

print("-" * 54)
print(f"{'R-squared':<28} {'0.2508':>12} {'0.2236':>12}")
print(f"{'Observations':<28} {'23,516':>12} {'31,350':>12}")