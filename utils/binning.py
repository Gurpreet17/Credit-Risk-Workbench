import os
import base64
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing as mp
from joblib import Parallel, delayed
from math import ceil

class binner:
    
    def __init__(self, dataset, target, binned_results={}, n_threshold=None, n_occurences=1, p_threshold=0.1, merge_threshold=None, sort_overload=None):
        self.dataset = dataset
        self.target = target
        self.n_threshold = n_threshold
        self.n_occurences = n_occurences
        self.p_threshold = p_threshold
        self.merge_threshold = merge_threshold
        self.sort_overload = sort_overload
        self.binned_results = binned_results
        columns = self.dataset.columns[self.dataset.columns != self.target]
        self.numeric_cols = self.dataset.loc[:, columns].select_dtypes(include=[np.number]).columns.tolist()
        self.categoric_cols = [col for col in self.dataset.loc[:, columns].select_dtypes(include=["object", "category"]).columns 
                                 if self.dataset.loc[:, col].nunique() < 100]
        
    def _create_label(self, row, column_name, is_categorical=False):
        """
        Create label for a bin based on interval or category.
        
        The binning algorithm stores intervals with special notation:
        - (start=inf, end=value) means "column >= value" (upper tail)
        - (start=value, end=-inf) means "column < value" (lower tail)
        - (start>end) means sorted interval after backwards storage
        
        This method must preserve these semantics for merge/split operations.
        """
        if is_categorical:
            return row['interval_start_include']
        
        start = row['interval_start_include']
        end = row['interval_end_exclude']
        
        # Handle missing values
        if pd.isna(start) and pd.isna(end):
            return "Missing"
        
        # Case 1: start=inf, end=finite → Upper tail "column >= end"
        if start == np.inf and end not in [np.inf, -np.inf, None] and pd.notna(end):
            end_str = f"{end:.2f}" if isinstance(end, float) else str(end)
            return f"{column_name} >= {end_str}"
        
        # Case 2: start=finite, end=-inf → Lower tail "column < start"
        if end == -np.inf and start not in [np.inf, -np.inf, None] and pd.notna(start):
            start_str = f"{start:.2f}" if isinstance(start, float) else str(start)
            return f"{column_name} < {start_str}"
        
        # Case 3: start=-inf, end=finite → Lower tail "column < end"
        if start == -np.inf and end not in [np.inf, -np.inf, None] and pd.notna(end):
            end_str = f"{end:.2f}" if isinstance(end, float) else str(end)
            return f"{column_name} < {end_str}"
        
        # Case 4: start=finite, end=inf → Upper tail "column >= start"
        if end == np.inf and start not in [np.inf, -np.inf, None] and pd.notna(start):
            start_str = f"{start:.2f}" if isinstance(start, float) else str(start)
            return f"{column_name} >= {start_str}"
        
        # Case 5: Both infinity
        if start in [np.inf, -np.inf] and end in [np.inf, -np.inf]:
            return "All values"
        
        # Case 6: Backwards interval (start > end) → Sort and create range
        if (start not in [np.inf, -np.inf, None] and end not in [np.inf, -np.inf, None] and
            pd.notna(start) and pd.notna(end) and start > end):
            low, high = end, start
            low_str = f"{low:.2f}" if isinstance(low, float) else str(low)
            high_str = f"{high:.2f}" if isinstance(high, float) else str(high)
            return f"{low_str} <= {column_name} < {high_str}"
        
        # Case 7: Normal interval (start <= end) → Standard range
        if (start not in [np.inf, -np.inf, None] and end not in [np.inf, -np.inf, None] and
            pd.notna(start) and pd.notna(end)):
            start_str = f"{start:.2f}" if isinstance(start, float) else str(start)
            end_str = f"{end:.2f}" if isinstance(end, float) else str(end)
            return f"{start_str} <= {column_name} < {end_str}"
        
        # Case 8: Fallback for any other pattern
        return "All values"

    def get_auc_gini(self, df):
        """
        Calculate AUC and Gini coefficient from binned WOE data.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain 'Non Target', 'Target', and 'Target Rate' columns.

        Returns
        -------
        tuple
            AUC (float), Gini coefficient (float)
        """
        # Sort bins by mean risk descending
        df_sorted = df.sort_values(by='Target Rate', ascending=False).reset_index(drop=True)
        # Prepare helper DataFrame
        df2 = pd.DataFrame()
        df2['Non Target'] = df_sorted['Non Target']
        df2['Target'] = df_sorted['Target']
        df2['prev_goods'] = df2['Non Target'].shift(1).fillna(0)
        df2['prev_goods_cum'] = df2['prev_goods'].cumsum()
        # Contributions
        df2['ranked_pairs_contribution'] = df2['Target'] * df2['prev_goods_cum']
        df2['tied_pairs_contribution'] = df2['Target'] * df2['Non Target']
        # Totals
        total_ranked_pairs_contribution = df2['ranked_pairs_contribution'].sum()
        total_tied_pairs_contribution = df2['tied_pairs_contribution'].sum()
        total_bad = df2['Target'].sum()
        total_good = df2['Non Target'].sum()
        # AUC and GINI
        AUC = 1 - round(((2 * total_ranked_pairs_contribution + total_tied_pairs_contribution) / (2 * total_bad * total_good)), 6)
        GINI = round(2 * AUC - 1, 6)
        return AUC, GINI
    
    def get_iv(self, df):
        """
        Calculate total Information Value (IV) from WOE-binned DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain 'iv' column.

        Returns
        -------
        float : Total IV rounded to 3 decimals.
        """
        iv = round(np.sum(df['iv']), 3)
        return iv
    
    def merge_bins(self, column, bins_index, woe_adjust=0.5):
        """
        Merge bins and recalculate WOE with continuity correction
        
        Parameters:
        -----------
        column : str
            Variable name
        bins_index : list
            Indices of bins to merge
        woe_adjust : float
            Continuity correction value (default 0.5)
        """
        bins_index = sorted(bins_index)
        
        print(f"[BINNER] Merging bins {bins_index} for {column}")
        
        # Validate indices exist
        max_index = self.binned_results[column].index.max()
        for idx in bins_index:
            if idx > max_index:
                raise ValueError(f"Index {idx} does not exist. Maximum index is {max_index}")
        
        # Get the merged interval bounds
        interval_start_include = self.binned_results[column].loc[bins_index[0], 'interval_start_include']
        interval_end_exclude = self.binned_results[column].loc[bins_index[-1], 'interval_end_exclude']
        
        # Get all rows being merged
        df_indexed = self.binned_results[column].loc[bins_index]
        
        if column in self.categoric_cols:
            # For categorical, combine categories
            merged_label = []
            for val in df_indexed['Value']:
                if isinstance(val, list):
                    merged_label.extend(val)
                else:
                    merged_label.append(val)
        else:
            # For numeric, create new range
            merged_label = self._create_label(
                pd.Series({'interval_start_include': interval_start_include, 
                        'interval_end_exclude': interval_end_exclude}),
                column_name=column, 
            )
        
        # Aggregate statistics
        size = int(df_indexed['Total'].sum())
        bads = int(df_indexed['Target'].sum())
        goods = int(df_indexed['Non Target'].sum())
        mean = bads / size if size > 0 else 0
        
        # Get TOTAL goods and bads from ALL bins (not just merged ones)
        total_goods = self.binned_results[column]['Non Target'].sum()
        total_bads = self.binned_results[column]['Target'].sum()
        
        # Calculate distributions with continuity correction
        dist_good = (goods + woe_adjust) / total_goods if total_goods > 0 else woe_adjust
        dist_bad = (bads + woe_adjust) / total_bads if total_bads > 0 else woe_adjust
        
        # Calculate WOE
        woe = np.log(dist_good / dist_bad)
        iv = (dist_good - dist_bad) * woe
        
        print(f"[BINNER] Merged bin stats: Total={size}, Target={bads}, Non-Target={goods}")
        print(f"[BINNER] Merged bin WOE: {woe:.4f}, IV: {iv:.4f}")
    
        # Create merged row
        merged_row = pd.DataFrame([{
            'Variable': self.binned_results[column]['Variable'].values[0],
            'Value': merged_label,
            'interval_end_exclude': interval_end_exclude,
            'interval_start_include': interval_start_include,
            'Total': size,
            'Target Rate': mean,
            'Target': bads,
            'Non Target': goods,
            'Non Target %': dist_good,
            'Target %': dist_bad,
            'WOE': woe,
            'iv': iv,
            'IV': 0,
            'AUC': 0,
            'GINI': 0
        }], index=[bins_index[0]])
        
        # Drop old rows and concat the merged row
        summary = self.binned_results[column].drop(bins_index)
        summary = pd.concat([summary, merged_row])
        
        # Recalculate IV, AUC, GINI for the entire variable
        summary['IV'] = self.get_iv(summary)
        summary['AUC'], summary['GINI'] = self.get_auc_gini(summary)
        
        # Sort and reset index
        result = summary.sort_index().reset_index(drop=True)
        
        print(f"[BINNER] After merge: {len(result)} bins remaining")
        print(f"[BINNER] New IV: {result['IV'].iloc[0]:.4f}")
        
        return result

    def split_bins(self, column, split_point, woe_adjust=0.5):
        """
        Split bins at a specified point with continuity correction
        
        Parameters:
        -----------
        column : str
            Column name
        split_point : float
            Point at which to split bins
        woe_adjust : float
            Continuity correction (default 0.5)
        """
        # Validate split_point
        try:
            split_point = float(split_point)
        except ValueError:
            raise ValueError("Not a number! Please enter a numeric value for the split.")
        
        # Current splits + new split
        current_splits = sorted(set(self.binned_results[column]['interval_start_include'].tolist() +
                                self.binned_results[column]['interval_end_exclude'].tolist()))
        current_splits = [x for x in current_splits if not pd.isna(x)]
        
        next_splits = sorted(set(current_splits + [split_point]))
    
        # Ensure column and target are numeric
        self.dataset[column] = self.dataset[column].astype(float)
        self.dataset[self.target] = self.dataset[self.target].astype(int)
        
        # Bin the original dataset
        self.dataset['bin'] = pd.cut(self.dataset[column], bins=next_splits, right=False)
        
        # Aggregate stats
        summary = self.dataset.groupby('bin').agg(
            Total=('bin', 'count'),
            Target=(self.target, 'sum')
        ).reset_index()
        summary['Non Target'] = summary['Total'] - summary['Target']
        summary['Target Rate'] = summary['Target'] / summary['Total']
        
        # Interval boundaries
        summary['interval_start_include'] = [b.left for b in summary['bin'].cat.categories]
        summary['interval_end_exclude'] = [b.right for b in summary['bin'].cat.categories]
        summary['Variable'] = column
        
        # Create labels for numeric bins
        summary['Value'] = summary.apply(lambda row: self._create_label(row, column_name=column, is_categorical=False), axis=1)
        
        # Distributions WITH continuity correction
        total_goods = summary['Non Target'].sum()
        total_bads = summary['Target'].sum()
        
        summary['Non Target %'] = (summary['Non Target'] + woe_adjust) / total_goods
        summary['Target %'] = (summary['Target'] + woe_adjust) / total_bads
        
        # WOE & IV
        summary['WOE'] = np.log(summary['Non Target %'] / summary['Target %'])
        summary['iv'] = (summary['Non Target %'] - summary['Target %']) * summary['WOE']
        summary['IV'] = self.get_iv(summary)
        summary['AUC'], summary['GINI'] = self.get_auc_gini(summary)
        
        nan_mask = self.dataset[column].isna()
        
        # Handle missing bin separately WITH continuity correction
        if nan_mask.any():
            size = nan_mask.sum()
            bads = self.dataset.loc[nan_mask, self.target].sum()
            goods = size - bads
            
            # Calculate WOE for missing bin with continuity correction
            missing_target_pct = (bads + woe_adjust) / total_bads
            missing_non_target_pct = (goods + woe_adjust) / total_goods
            missing_woe = np.log(missing_non_target_pct / missing_target_pct)
            missing_iv = (missing_non_target_pct - missing_target_pct) * missing_woe

            summary = pd.concat(
                [summary, pd.DataFrame([{
                    'Variable': column,
                    'Value': 'Missing',
                    'interval_end_exclude': np.nan,
                    'interval_start_include': np.nan,
                    'Total': size,
                    'Target Rate': bads / size if size > 0 else 0,
                    'Target': bads,
                    'Non Target': goods,
                    'Non Target %': missing_non_target_pct,
                    'Target %': missing_target_pct,
                    'WOE': missing_woe,
                    'iv': missing_iv
                }])],
                ignore_index=True
            )
        
        # Final columns
        summary = summary[['Variable', 'Value', 'interval_end_exclude', 'interval_start_include',
                        'Total', 'Target Rate', 'Target', 'Non Target', 'Non Target %', 'Target %',
                        'WOE', 'iv', 'IV', 'AUC', 'GINI']]
        
         
        summary = summary.sort_values('interval_start_include', ascending=True, na_position='last')
        summary = summary.reset_index(drop=True)
        return summary
    
    def save_binnings(self, column, df):
        """Save binning results for a column"""
        self.binned_results[column] = df
        
    def batch_woe_binning(self):
        """Batch WOE binning for all variables"""
        nprocs = 1
        
        if self.n_threshold is None:
            self.n_threshold = ceil(self.dataset.shape[0] / 20)
       
        parallel = Parallel(n_jobs=nprocs, verbose=5)

        # Use instance variables (single source of truth)
        numeric_list = (Parallel(n_jobs=-1)(
            delayed(self.woe_numeric_binning)(col) for col in self.numeric_cols
        ) if self.numeric_cols else [])
        
        categoric_list = (Parallel(n_jobs=-1)(
            delayed(self.woe_categoric_binning)(col) for col in self.categoric_cols
        ) if self.categoric_cols else [])

        del parallel
        numeric_key = {i.Variable[0]: i for i in numeric_list}
        categoric_key = {i.Variable[0]: i for i in categoric_list}
        self.binned_results = {**numeric_key, **categoric_key}
        
    def woe_categoric_binning(self, column, woe_adjust=0.5):
        """
        WOE binning for categorical variables with continuity correction
        
        Parameters:
        -----------
        column : str
            Categorical column name
        woe_adjust : float
            Continuity correction value (default 0.5)
        """
        df = self.dataset.loc[:, [column, self.target]].copy()
        df.loc[:, column] = df.loc[:, column].fillna('MISSING')
        
        summary = df.groupby(column)[self.target].agg(["mean", "size", "std"])
        summary = summary[["mean", "size", "std"]]
        summary = summary.reset_index()
        summary["del_flag"] = 0 
        summary["std"] = summary["std"].fillna(0)
        
        woe_summary = summary[[column, "size", "mean"]].copy()
        woe_summary.columns = ["interval_start_include", "Total", "Target Rate"]
        woe_summary.loc[:, "interval_end_exclude"] = woe_summary["interval_start_include"]
        woe_summary["Variable"] = column
        woe_summary['Value'] = woe_summary['interval_start_include']
        
        woe_summary["Target"] = woe_summary["Target Rate"] * woe_summary["Total"]
        woe_summary["Non Target"] = woe_summary["Total"] - woe_summary["Target"]
        
        total_target = np.sum(woe_summary["Target"])
        total_non_target = np.sum(woe_summary["Non Target"])
        
        # Apply continuity correction
        woe_summary["Target %"] = (woe_summary["Target"] + woe_adjust) / total_target
        woe_summary["Non Target %"] = (woe_summary["Non Target"] + woe_adjust) / total_non_target
        woe_summary["WOE"] = np.log(woe_summary["Non Target %"] / woe_summary["Target %"])
        woe_summary["iv"] = (woe_summary["Non Target %"] - woe_summary["Target %"]) * woe_summary["WOE"]
        
        woe_summary['IV'] = self.get_iv(woe_summary)
        woe_summary['AUC'], woe_summary['GINI'] = self.get_auc_gini(woe_summary)
        
        # Reorder columns
        woe_summary = woe_summary[["Variable", "Value", "interval_end_exclude", "interval_start_include", 
                                    "Total", "Target Rate", "Target", "Non Target", "Non Target %", 
                                    "Target %", "WOE", "iv", "IV", "AUC", "GINI"]]
        
        woe_summary = woe_summary.sort_values('WOE', ascending=True)
        woe_summary.reset_index(drop=True, inplace=True)
        return woe_summary
    
    def woe_numeric_binning(self, column, woe_adjust=0.5):
        """
        WOE binning for numeric variables with continuity correction
        
        Parameters:
        -----------
        column : str
            Numeric column name
        woe_adjust : float
            Continuity correction value (default 0.5)
        """
        df = self.dataset.loc[:, [column, self.target]]
        sorted_dataset = df.sort_values(by=[column])
        size = sorted_dataset.shape[0]

        if sorted_dataset[:int(size / 4)][self.target].sum() > sorted_dataset[int(size * 3 / 4):][self.target].sum():
            order = True
            interval_end = np.inf
        else:
            order = False
            interval_end = -np.inf

        summary = df.dropna().groupby([column]).agg(["mean", "size", "std"])
        summary.columns = summary.columns.droplevel(level=0)
        summary = summary[["mean", "size", "std"]]
        summary = summary.reset_index()
        summary["del_flag"] = 0
        summary["std"] = summary["std"].fillna(0)
        summary = summary.sort_values(by=[column], ascending=(self.sort_overload or order)).reset_index(drop=True)

        # Monotonic binning algorithm
        while True:
            i = 0
            summary = summary[summary.del_flag == 0]
            summary = summary.reset_index(drop=True)

            while True:
                j = i + 1
                if j >= len(summary):
                    break

                if summary.iloc[j]['mean'] < summary.iloc[i]['mean']:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = summary.iloc[j]['size'] + summary.iloc[i]['size']
                        m = (summary.iloc[j]['size'] * summary.iloc[j]['mean'] +
                            summary.iloc[i]['size'] * summary.iloc[i]['mean']) / n

                        if n == 2:
                            s = np.std([summary.iloc[j]['mean'], summary.iloc[i]['mean']])
                        else:
                            s = np.sqrt((summary.iloc[j]['size'] * ((summary.iloc[j]['std']) ** 2) +
                                        summary.iloc[i]['size'] * ((summary.iloc[i]['std']) ** 2)) / n)

                        summary.loc[i, "size"] = n
                        summary.loc[i, "mean"] = m
                        summary.loc[i, "std"] = s
                        summary.loc[j, "del_flag"] = 1

                        j = j + 1

                        if j >= len(summary):
                            break
                        if summary.loc[j, "mean"] < summary.loc[i, "mean"]:
                            i = j
                            break
                if j >= len(summary):
                    break

            dels = np.sum(summary["del_flag"])
            if dels == 0:
                break

        # Statistical merging
        while True:
            summary["next_mean"] = summary["mean"].shift(-1)
            summary["next_size"] = summary["size"].shift(-1)
            summary["next_std"] = summary["std"].shift(-1)

            summary["updated_size"] = summary["next_size"] + summary["size"]
            summary["updated_mean"] = (summary["next_mean"] * summary["next_size"] +
                                    summary["mean"] * summary["size"]) / summary["updated_size"]

            summary["updated_std"] = (summary["next_size"] * summary["next_std"] ** 2 +
                                    summary["size"] * summary["std"] ** 2) / (summary["updated_size"] - 2)

            summary["z_value"] = (summary["mean"] - summary["next_mean"]) / np.sqrt(
                summary["updated_std"] * (1 / summary["size"] + 1 / summary["next_size"]))

            summary["p_value"] = 1 - stats.norm.cdf(summary["z_value"])

            condition = (summary["size"] < self.n_threshold) | (summary["next_size"] < self.n_threshold) | (
                    summary["mean"] * summary["size"] < self.n_occurences) | (
                                summary["next_mean"] * summary["next_size"] < self.n_occurences)

            summary.loc[condition, 'p_value'] = summary.loc[condition, 'p_value'] + 1

            summary["p_value"] = summary.apply(
                lambda row: row["p_value"] + 1 if (row["size"] < self.n_threshold) | (row["next_size"] < self.n_threshold) |
                                                (row["mean"] * row["size"] < self.n_occurences) |
                                                (row["next_mean"] * row["next_size"] < self.n_occurences)
                else row["p_value"], axis=1)

            vals = pd.to_numeric(summary["p_value"], errors="coerce").dropna().tolist()
            max_p = max(vals, default=None)
            if max_p is None:
                break
            row_of_maxp = summary["p_value"].idxmax()
            pos = summary.index.get_loc(row_of_maxp)
            if pos + 1 >= len(summary):
                break
            row_delete = summary.index[pos + 1]

            if max_p > self.p_threshold:
                summary = summary.drop(summary.index[row_delete])
                summary = summary.reset_index(drop=True)
            else:
                break

            summary["mean"] = summary.apply(lambda row: row["updated_mean"] if row["p_value"] == max_p else row["mean"],
                                            axis=1)
            summary["size"] = summary.apply(lambda row: row["updated_size"] if row["p_value"] == max_p else row["size"],
                                            axis=1)
            summary["std"] = summary.apply(
                lambda row: np.sqrt(row["updated_std"]) if row["p_value"] == max_p else row["std"], axis=1)

        woe_summary = summary[[column, "size", "mean"]].copy()
        woe_summary.columns = ["interval_start_include", "Total", "Target Rate"]
        woe_summary["interval_start_include"] = woe_summary["interval_start_include"].astype("float64")
        woe_summary.loc[:, "interval_end_exclude"] = woe_summary.loc[:, "interval_start_include"].shift(-1).fillna(interval_end)
        woe_summary.loc[woe_summary.index[0], "interval_start_include"] = interval_end * -1
        woe_summary.loc[:, "Variable"] = column
        woe_summary = woe_summary.loc[:, ["Variable", "interval_end_exclude", "interval_start_include", "Total", "Target Rate"]]

        # Calculate Target/Non-Target BEFORE handling missing
        woe_summary["Target"] = woe_summary["Target Rate"] * woe_summary["Total"]
        woe_summary["Non Target"] = woe_summary["Total"] - woe_summary["Target"]

        # Get totals (BEFORE adding missing row)
        total_goods = np.sum(woe_summary["Non Target"])
        total_bads = np.sum(woe_summary["Target"])

        # Apply continuity correction to all regular bins
        woe_summary["Non Target %"] = (woe_summary["Non Target"] + woe_adjust) / total_goods
        woe_summary["Target %"] = (woe_summary["Target"] + woe_adjust) / total_bads
        woe_summary["WOE"] = np.log(woe_summary["Non Target %"] / woe_summary["Target %"])
        woe_summary["iv"] = (woe_summary["Non Target %"] - woe_summary["Target %"]) * woe_summary["WOE"]

        woe_summary['IV'] = self.get_iv(woe_summary)
        woe_summary['AUC'], woe_summary['GINI'] = self.get_auc_gini(woe_summary)

        # Handle missing values WITH continuity correction
        if df[column].isna().sum() > 0:
            missing_data = df[df[column].isna()]
            missing_total = len(missing_data)
            missing_target = missing_data[self.target].sum()
            missing_non_target = missing_total - missing_target
            missing_target_rate = missing_target / missing_total if missing_total > 0 else 0
            
            # Apply continuity correction to missing bin
            missing_target_pct = (missing_target + woe_adjust) / total_bads
            missing_non_target_pct = (missing_non_target + woe_adjust) / total_goods
            missing_woe = np.log(missing_non_target_pct / missing_target_pct)
            missing_iv = (missing_non_target_pct - missing_target_pct) * missing_woe
            
            iv_val = woe_summary['IV'].iloc[0]
            auc_val = woe_summary['AUC'].iloc[0]
            gini_val = woe_summary['GINI'].iloc[0]
            
            missing_row = pd.DataFrame([{
                'Variable': column,
                'interval_end_exclude': np.nan,
                'interval_start_include': np.nan,
                'Total': missing_total,
                'Target Rate': missing_target_rate,
                'Target': missing_target,
                'Non Target': missing_non_target,
                'Non Target %': missing_non_target_pct,
                'Target %': missing_target_pct,
                'WOE': missing_woe,
                'iv': missing_iv,
                'IV': iv_val,
                'AUC': auc_val,
                'GINI': gini_val
            }])
            
            woe_summary = pd.concat([woe_summary, missing_row], ignore_index=True)

        # Create labels (do this AFTER adding missing row)
        woe_summary['Value'] = woe_summary.apply(
            lambda row: 'Missing' if pd.isna(row['interval_start_include']) 
            else self._create_label(row, column_name=column, is_categorical=False), 
            axis=1
        )

        # Merge bins if threshold is set
        if self.merge_threshold:
            while True:
                non_missing_mask = ~pd.isna(woe_summary['interval_start_include'])
                non_missing_woe = woe_summary[non_missing_mask]
                
                if non_missing_woe.shape[0] <= 1:
                    break
                    
                merged = False
                for i in range(non_missing_woe.shape[0] - 1):
                    idx_i = non_missing_woe.index[i]
                    idx_j = non_missing_woe.index[i + 1]
                    
                    woe_i = woe_summary.loc[idx_i, 'WOE']
                    woe_j = woe_summary.loc[idx_j, 'WOE']
                    
                    if abs(woe_i) > 0:
                        if abs(abs(woe_i) - abs(woe_j)) / abs(woe_i) <= self.merge_threshold:
                            woe_summary = self.merge_bins(column, [idx_i, idx_j], woe_adjust=woe_adjust)
                            merged = True
                            break
                
                if not merged:
                    break
        
        # Reorder columns
        woe_summary = woe_summary[[
            'Variable', 'Value', 'interval_end_exclude', 'interval_start_include',
            'Total', 'Target Rate', 'Target', 'Non Target', 'Non Target %', 'Target %',
            'WOE', 'iv', 'IV', 'AUC', 'GINI'
        ]]
        
        # Sort (Missing stays at end due to NaN)
        woe_summary = woe_summary.sort_values(
            by='interval_start_include', 
            ascending=True, 
            na_position='last'
        )
        woe_summary = woe_summary.reset_index(drop=True)
        
        return woe_summary

    def apply_bins(self, dataset=None, dict_woe=None):
        """
        Apply WOE binning to a dataset
        
        Parameters:
        -----------
        dataset : pd.DataFrame, optional
            Data to transform. If None, uses self.dataset
        dict_woe : dict, optional
            Binning rules. If None, uses self.binned_results
      
        Returns:
        --------
        pd.DataFrame
            WOE-transformed dataset
        """
        # Use provided parameters or fall back to instance attributes
        if dataset is None:
            dataset = self.dataset
        if dict_woe is None:
            dict_woe = self.binned_results
        
        # Create output DataFrame
        woe_dataset = pd.DataFrame(index=dataset.index)
        
        # Get variables to process - ONLY process variables in BOTH dataset and dict_woe
        available_vars = [var for var in dict_woe.keys() if var in dataset.columns]
        
        for var in available_vars:
            # Get binning rules for this variable
            df_col = dict_woe[var]
            column = str(var)
            
            # Use standardized naming
            if var in self.categoric_cols:
                # CATEGORICAL VARIABLE
                cat_map = {}
                for idx, row in df_col.iterrows():
                    woe_value = pd.to_numeric(row["WOE"], errors="coerce")
                    
                    # Handle merged categories (list)
                    if isinstance(row["Value"], list):
                        for category in row["Value"]:
                            cat_map[str(category)] = woe_value
                    else:
                        cat_map[str(row["Value"])] = woe_value
                
                # Default WOE for unseen categories
                default_woe = cat_map.get('OTHER', 0.0)
                
                s = dataset[column].fillna("MISSING").astype(str)
                mapped = s.map(cat_map)
                
                # Handle unseen categories
                unseen_mask = mapped.isna()
                if unseen_mask.any():
                    mapped = mapped.fillna(default_woe)
                    
                    unseen_count = unseen_mask.sum()
                    if unseen_count > 0:
                        unseen_values = s[unseen_mask].unique()
                        if unseen_count / len(s) > 0.01:  # Warn if >1% unseen
                            print(f"⚠️ '{column}': {unseen_count} ({unseen_count/len(s)*100:.2f}%) unseen categories → WOE={default_woe:.4f}")
                            print(f"   Examples: {list(unseen_values)[:5]}")
                
                woe_dataset[column] = mapped
            
            else:
                # NUMERIC VARIABLE
                # Get WOE for missing values
                missing_rows = df_col[df_col["interval_start_include"].isna() & df_col["interval_end_exclude"].isna()]
                if len(missing_rows) > 0:
                    missing_woe = missing_rows["WOE"].iloc[0]
                else:
                    missing_woe = None
                
                # Drop missing bin row
                df_col_numeric = df_col.dropna(subset=["interval_start_include", "interval_end_exclude", "WOE"]).copy()
                
                if len(df_col_numeric) == 0:
                    woe_dataset[column] = 0.0
                    continue
                
                # Sort and prepare edges
                df_col_sorted = df_col_numeric.sort_values(by="interval_start_include")
                ends_sorted = df_col_sorted["interval_end_exclude"].tolist()
                labels_sorted = pd.to_numeric(df_col_sorted["WOE"], errors="coerce").tolist()

                # Build edges
                edges = [-np.inf] + ends_sorted[:-1] + [np.inf]
                edges = np.array(edges, dtype=float)
                edges = np.unique(edges)
                
                if len(edges) < 2:
                    woe_dataset[column] = 0.0
                    continue

                # Align labels to bins count
                labels_aligned = labels_sorted[:len(edges) - 1]

                # Get series and identify NaN
                series = pd.to_numeric(dataset[column], errors="coerce")
                nan_mask = series.isna()
                
                # Apply pd.cut to non-NaN values
                binned_values = pd.cut(series, bins=edges, include_lowest=True, right=False, labels=labels_aligned)
                
                # Convert to float
                woe_dataset[column] = pd.to_numeric(binned_values, errors='coerce')
                
                # Map NaN values to their WOE
                if missing_woe is not None:
                    woe_dataset.loc[nan_mask, column] = missing_woe
        
        return woe_dataset

    def woe_plot(self, column, figsize=(18, 8), rotate_xticks=45):
        """
        Plot Weight of Evidence vs bins with bin counts and target/non-target proportions
        """
        plot_df = self.binned_results[column].copy()
        
        var = plot_df.at[0, 'Variable']
        iv_value = round(plot_df['IV'].iloc[0], 3)
        gini_value = round(plot_df['GINI'].iloc[0], 3)
        
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel(var)
        ax1.set_ylabel("# Observations")
        
        x_labels = plot_df['Value'].astype(str)
        ax1.bar(x_labels, plot_df['Target'], color='red', label="Target")
        ax1.bar(x_labels, plot_df['Non Target'], color='green', bottom=plot_df['Target'], label="Non Target")
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("WOE")
        ax2.plot(x_labels, plot_df['WOE'], marker='o', linestyle='--', color='black', label="WOE")
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
        
        plt.title(f"{var}\nIV = {iv_value} | GINI = {gini_value}", fontsize=14)
        
        for tick in ax1.get_xticklabels():
            tick.set_rotation(rotate_xticks)
        
        plt.tight_layout()
        plt.show()
        plt.close()