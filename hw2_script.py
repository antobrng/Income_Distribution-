import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
ECON 473 - Homework 2 Script (Robust Weights)
"""

""" 
helper functions for question 4 
"""
def weighted_variance(series, weights):
    if len(series) == 0: return np.nan
    average = np.average(series, weights=weights)
    variance = np.average((series - average)**2, weights=weights)
    return variance 

def weighted_gini(series, weights):
    series = np.array(series)
    weights = np.array(weights)
    sorted_indices = np.argsort(series)
    series = series[sorted_indices]
    weights = weights[sorted_indices]
    
    cumsum_weights = np.cumsum(weights)
    sum_weights = cumsum_weights[-1]
    
    cumsum_weighted_series = np.cumsum(series * weights)
    sum_weighted_series = cumsum_weighted_series[-1]
    
    x = cumsum_weights / sum_weights
    y = cumsum_weighted_series / sum_weighted_series
    
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)
    
    auc = np.trapezoid(y, x)
    return 1 - 2 * auc

def weighted_percentile(data, weights, percentile):
    if len(data) == 0: return np.nan
    sorted_indices = np.argsort(data)
    data = np.array(data)[sorted_indices]
    weights = np.array(weights)[sorted_indices]
    
    cumsum_weights = np.cumsum(weights)
    sum_weights = cumsum_weights[-1]
    
    percentile_positions = 100 * cumsum_weights / sum_weights
    idx = np.searc_hwsorted(percentile_positions, percentile)
    
    if idx >= len(data): return data[-1]
    return data[idx]

def weighted_corr(x, y, w):
    def weighted_mean(arr, w):
        return np.sum(arr * w) / np.sum(w)

    def weighted_cov(x, y, w):
        mx = weighted_mean(x, w)
        my = weighted_mean(y, w)
        return np.sum(w * (x - mx) * (y - my)) / np.sum(w)

    cov_xy = weighted_cov(x, y, w)
    var_x = weighted_cov(x, x, w)
    var_y = weighted_cov(y, y, w)
    
    if var_x == 0 or var_y == 0: return np.nan
    return cov_xy / np.sqrt(var_x * var_y)


# Question 1
# Load datasets
print("Loading data...")
df_1982 = pd.read_csv('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/data/scf-13M00004-E-1982-ind_F1.csv')
df_1990 = pd.read_csv('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/data/scf-13M00004-E-1990-ind_F1.csv')
df_2016 = pd.read_csv('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/data/CIS-72M0003-E-2016_F1.csv')
df_2018 = pd.read_csv('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/data/CIS-72M0003-E-2018_F1.csv')

datasets = {1982: df_1982, 1990: df_1990, 2016: df_2016, 2018: df_2018}

# Clean 2016/2018 invalid codes
for year in [2016, 2018]:
    df = datasets[year]
    df['USHRWK'] = np.where(df['USHRWK'] >= 9996, np.nan, df['USHRWK'])
    df['WKSEM'] = np.where(df['WKSEM'] >= 96, np.nan, df['WKSEM'])
    for col in ['WGSAL', 'ATINC', 'TTINC']:
        if col in df.columns:
            df[col] = np.where(df[col] >= 99999996, np.nan, df[col])
    datasets[year] = df

for year, df in datasets.items():
    print(f"Processing initial variables for {year}...")
    
    if year in [1982, 1990]:
        wage_var, hours_var, weeks_var = 'WAGSAL', 'HRSWRK', 'WKSWRK'
        target_weight = 'WEIGHT'
    else:
        wage_var, hours_var, weeks_var = 'WGSAL', 'USHRWK', 'WKSEM'
        target_weight = 'FWEIGHT'

    # handle wieghts for 1982
    if target_weight in df.columns:
        df['weight'] = df[target_weight].fillna(0)
        if df['weight'].sum() == 0:
            print(f"  WARNING: {target_weight} column empty. Setting weights to 1 (Unweighted).")
            df['weight'] = 1
    else:
        print(f"  NOTICE: {target_weight} not found for {year}. Setting weights to 1 (Unweighted).")
        df['weight'] = 1

    df = df[(df[hours_var] > 0) & (df[weeks_var] > 0)].copy()
    df['hourly_wage'] = df[wage_var] / (df[hours_var] * df[weeks_var])
    
    datasets[year] = df


# question 2
# CPI values
cpi = {1982: 40.07, 1990: 57.23, 2016: 93.72, 2018: 97.37}
# plot cpi evolution
cpi_data = pd.DataFrame({'year': list(cpi.keys()), 'cpi': list(cpi.values())})
plt.figure(figsize=(10, 6))
plt.plot(cpi_data['year'], cpi_data['cpi'], 'b-', linewidth=2, label='CPI')
plt.scatter(cpi_data['year'], cpi_data['cpi'], color='blue', s=100, zorder=5)
plt.title('Evolution of CPI (1982-2018)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Consumer Price Index', fontsize=12)
plt.xticks([1982, 1990, 2016, 2018])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/plots/cpi_evolution.png', dpi=300)
plt.show()

# deflate wages 
for y, df in datasets.items():
    df['hourly_real_wage'] = (df['hourly_wage'] / cpi[y]) * 100
    datasets[y] = df


# question 3
unfiltered_datasets = {year: df.copy() for year, df in datasets.items()}

min_wage_thresholds = [3.7375*0.5, 5.0500001*0.5, 11.3*0.5, 13.4125*0.5]
years = [1982, 1990, 2016, 2018]

for year, wage in zip(years, min_wage_thresholds):
    df = datasets[year]
    
    print("\n" + "="*65)
    print(f"Table 1: \n Sample Selection in the {year} Dataset")
    print("="*65)

    if year in [1982, 1990]:
        hours_var, weeks_var, age_var = 'HRSWRK', 'WKSWRK', 'AGE'
    else:
        hours_var, weeks_var, age_var = 'USHRWK', 'WKSEM', 'AGEGP'

    n_total = len(df)

    # Wage & Work Filter
    df_clean = df[
        (df['hourly_wage'].notna()) & (df['hourly_wage'] >= wage) & 
        (df[hours_var] > 0) & (df[weeks_var] > 0)
    ]
    n_after_wage = len(df_clean)
    drop1 = n_total - n_after_wage

    # Age Filter
    if year in [1982, 1990]:
        df_clean = df_clean[(df_clean[age_var] >= 25) & (df_clean[age_var] <= 60)]
    else:
        df_clean = df_clean[(df_clean[age_var] >= 7) & (df_clean[age_var] <= 14)]

    n_after_age = len(df_clean)
    drop2 = n_after_wage - n_after_age
    # table 1 replication
    summary = pd.DataFrame({
        'Step': ['Initial data', 'After wage filtering', 'After age filtering'],
        'Observations': [n_total, n_after_wage, n_after_age],
        'Dropped': [0, drop1, drop2],
        'Cumulative Dropped': [0, drop1, drop1 + drop2]
    })
    print(summary.to_string(index=False))
    print("="*65)

    datasets[year] = df_clean


# question 4 
# part A - log var and gini for total income, after-tax income, and wages
print("\nCalculating Weighted Statistics...")
vars_map = {
    1982: {'Tot': 'TOTINC', 'Tax': 'INCAFTTX', 'Wages': 'WAGSAL'},
    1990: {'Tot': 'TOTINC', 'Tax': 'INCAFTTX', 'Wages': 'WAGSAL'},
    2016: {'Tot': 'TTINC', 'Tax': 'ATINC', 'Wages': 'WGSAL'},
    2018: {'Tot': 'TTINC', 'Tax': 'ATINC', 'Wages': 'WGSAL'}
}

log_vars = {'Total': [], 'Tax': [], 'Wages': []}
ginis = {'Total': [], 'Tax': [], 'Wages': []}

for year in years:
    df = datasets[year]
    cols = vars_map[year]
    w = df['weight']

    for key, col in zip(['Total', 'Tax', 'Wages'], [cols['Tot'], cols['Tax'], cols['Wages']]):
        s = df[col]
        # stricly positive income and weights 
        mask_pos = (s > 0) & (w > 0)
        mask_gin = (s >= 0) & (w > 0)
        # Log Var
        if mask_pos.sum() > 0:
            log_vars[key].append(weighted_variance(np.log(s[mask_pos]), w[mask_pos]))
        else:
            log_vars[key].append(np.nan) 
        # Gini
        if mask_gin.sum() > 0:
            ginis[key].append(weighted_gini(s[mask_gin], w[mask_gin]))
        else:
            ginis[key].append(np.nan)

# variance of log
plt.figure(figsize=(10, 6))
plt.plot(years, log_vars['Total'], 'b-o', label='Total Income')
plt.plot(years, log_vars['Tax'], 'g--s', label='After-Tax')
plt.plot(years, log_vars['Wages'], 'r-.^', label='Wages')
plt.title('Weighted Variance of Log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(1980, 2021, 5))
plt.savefig('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/plots/weighted_log_variance.png', dpi=300)
plt.show()

# gini 
plt.figure(figsize=(10, 6))
plt.plot(years, ginis['Total'], 'b-o', label='Total Income')
plt.plot(years, ginis['Tax'], 'g--s', label='After-Tax')
plt.plot(years, ginis['Wages'], 'r-.^', label='Wages')
plt.title('Weighted Gini Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(1980, 2021, 5))
plt.savefig('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/plots/weighted_gini.png', dpi=300)
plt.show()


# part B - annual wage, hourly wage, annual hours, and their correlation
print("\nCalculating Weighted Labor Stats (Part B)...")
groups = {'All': None, 'Men': 1, 'Women': 2}
results_part_b = {g: {'v_aw': [], 'v_hw': [], 'v_ah': [], 'corr': []} for g in groups}

for year in years:
    df_all = datasets[year]
    
    if year in [1982, 1990]:
        c_aw, c_hw, c_ww = 'WAGSAL', 'HRSWRK', 'WKSWRK'
    else:
        c_aw, c_hw, c_ww = 'WGSAL', 'USHRWK', 'WKSEM'
        
    for group_name, gender_code in groups.items():
        if gender_code: 
            df_group = df_all[df_all['SEX'] == gender_code]
        else: 
            df_group = df_all
        
        ann_hrs = df_group[c_hw] * df_group[c_ww]
        hr_wage = df_group['hourly_wage']
        ann_wage = df_group[c_aw]
        w = df_group['weight']
        
        mask = (ann_wage > 0) & (hr_wage > 0) & (ann_hrs > 0) & (w > 0)
        d_sub = df_group[mask]
        wt = w[mask]
        
        if len(d_sub) > 0:
            ln_aw = np.log(ann_wage[mask])
            ln_hw = np.log(hr_wage[mask])
            ln_ah = np.log(ann_hrs[mask])
            
            results_part_b[group_name]['v_aw'].append(weighted_variance(ln_aw, wt))
            results_part_b[group_name]['v_hw'].append(weighted_variance(ln_hw, wt))
            results_part_b[group_name]['v_ah'].append(weighted_variance(ln_ah, wt))
            results_part_b[group_name]['corr'].append(weighted_corr(ln_hw, ln_ah, wt))
        else:
            for k in ['v_aw', 'v_hw', 'v_ah', 'corr']: results_part_b[group_name][k].append(np.nan)

# 3 plorts for Part B (all, men, women)
for group_name in groups:
    dat = results_part_b[group_name]
    plt.figure(figsize=(10, 6))
    plt.plot(years, dat['v_aw'], 'b-o', label='Var Log Annual Wage')
    plt.plot(years, dat['v_hw'], 'g--s', label='Var Log Hourly Wage')
    plt.plot(years, dat['v_ah'], 'r-.^', label='Var Log Annual Hours')
    plt.plot(years, dat['corr'], 'k:x', label='Corr(Wage, Hours)')
    plt.title(f'Weighted Labor Dynamics: {group_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1980, 2021, 5))
    plt.savefig(f'/Users/antoninberanger/Documents/ECON473/homeworks/HW2/plots/weighted_part_b_{group_name}.png', dpi=300)
    plt.show()


# part C - Weighted Percentiles
print("\nCalculating Weighted Percentiles (Part C)...")
p5, p50, p95 = [], [], []

for year in years:
    df = datasets[year]
    wage = df['hourly_real_wage']
    w = df['weight']
    
    mask = (wage.notna()) & (w > 0)
    if mask.sum() > 0:
        p5.append(weighted_percentile(wage[mask], w[mask], 5))
        p50.append(weighted_percentile(wage[mask], w[mask], 50))
        p95.append(weighted_percentile(wage[mask], w[mask], 95))
    else:
        for l in [p5, p50, p95]: l.append(np.nan)

plt.figure(figsize=(10, 6))
plt.plot(years, p95, 'b-o', label='P95')
plt.plot(years, p50, 'k--s', label='Median')
plt.plot(years, p5, 'r-.^', label='P5')
plt.title('Weighted Real Hourly Wage Percentiles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(1980, 2021, 5))
plt.savefig('/Users/antoninberanger/Documents/ECON473/homeworks/HW2/plots/weighted_percentiles.png', dpi=300)
plt.show()

print("\nDone. All calculations and plots generated.")