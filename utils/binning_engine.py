"""
Binning Engine Wrapper
Wraps the existing binner class for integration with PyQt6
"""
import pandas as pd
import numpy as np
from utils.binning import binner
from math import ceil


class BinningEngineWrapper:
    """Wrapper around existing binner class for UI integration"""
    
    def __init__(self, train_data, target_variable):
        """
        Initialize binning engine wrapper
        Args:
            train_data: Training DataFrame
            target_variable: Name of target column (must be 0/1)
        """
        self.train_data = train_data.copy()
        self.target_variable = target_variable
        
        # Calculate n_threshold (minimum observations per bin)
        n_threshold = max(ceil(len(train_data) / 20), 5)  # At least 5 observations
        
        # Create binner instance - THIS IS THE SINGLE SOURCE OF TRUTH
        self.binner = binner(
            dataset=self.train_data,
            target=self.target_variable,
            binned_results={},
            n_threshold=n_threshold,
            n_occurences=1,
            p_threshold=0.1,
            merge_threshold=None,
            sort_overload=None
        )
        
        self.binned_results = {}
        self.variable_metrics = {}
    
    @property
    def numeric_cols(self):
        """
        Get numeric columns from binner
        
        Returns:
            list: List of numeric column names
        """
        return self.binner.numeric_cols
    
    @property
    def categoric_cols(self):
        """
        Get categorical columns from binner
        
        Returns:
            list: List of categorical column names
        """
        return self.binner.categoric_cols
    

    def calculate_all_metrics(self):
        """Extract IV, AUC, and Gini from binned results"""
        for variable, result_df in self.binned_results.items():
            try:
                # Extract metrics (already calculated by binner)
                iv = result_df['IV'].iloc[0] if 'IV' in result_df.columns else 0
                auc = result_df['AUC'].iloc[0] if 'AUC' in result_df.columns else 0.5
                gini = result_df['GINI'].iloc[0] if 'GINI' in result_df.columns else 0
                
                # Determine type
                is_categorical = variable in self.categoric_cols
                var_type = 'categoric' if is_categorical else 'numeric'
                
                self.variable_metrics[variable] = {
                    'IV': iv,
                    'AUC': auc,
                    'Gini': gini,
                    'type': var_type
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {variable}: {str(e)}")
                self.variable_metrics[variable] = {
                    'IV': 0,
                    'AUC': 0.5,
                    'Gini': 0,
                    'type': 'unknown'
                }
    
    def merge_bins(self, variable, bin_indices):
        """
        Merge specified bins for a variable
        
        Args:
            variable: Variable name
            bin_indices: List of bin indices to merge (e.g., [0, 1])
        """
        if variable not in self.binned_results:
            raise ValueError(f"Variable {variable} not found in binned results")
        
        try:
            # Use binner's merge_bins method
            merged_df = self.binner.merge_bins(variable, bin_indices)
            
            # Update stored results
            self.binned_results[variable] = merged_df
            self.binner.binned_results[variable] = merged_df
            
            # Recalculate metrics
            self.calculate_all_metrics()
            
            return merged_df
            
        except Exception as e:
            raise Exception(f"Error merging bins: {str(e)}")
    
    def split_bins(self, variable, split_value):
        """
        Split a numeric bin at specified value
        
        Args:
            variable: Variable name
            split_value: Value at which to split
        """
        if variable not in self.binned_results:
            raise ValueError(f"Variable {variable} not found in binned results")
        
        # Check if numeric
        is_categorical = variable in self.categoric_cols
        
        if is_categorical:
            raise ValueError(f"Cannot split categorical variable {variable}")
        
        try:
            # Use binner's split_bins method
            split_df = self.binner.split_bins(variable, split_value)
            
            # Update stored results
            self.binned_results[variable] = split_df
            self.binner.binned_results[variable] = split_df
            
            # Recalculate metrics
            self.calculate_all_metrics()
            
            return split_df
            
        except Exception as e:
            raise Exception(f"Error splitting bins: {str(e)}")
    
    def save_binnings(self):
        """Save current binning configuration"""
        import copy
        return copy.deepcopy(self.binned_results)
    
    def get_sorted_variables(self):
        """Get variables sorted by IV (descending)"""
        sorted_vars = sorted(
            self.variable_metrics.items(),
            key=lambda x: x[1]['IV'],
            reverse=True
        )
        return sorted_vars