"""
Universal data loader utility
"""
import os
import pandas as pd


def load_data_universal(file_path):
    """
    Universal data loader - handles any common data format
    
    Supported formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls, .xlsm, .xlsb)
    - JSON (.json)
    - Parquet (.parquet, .pq)
    - Feather (.feather)
    - SAS (.sas7bdat, .xpt)
    - STATA (.dta)
    - SPSS (.sav)
    - HDF5 (.h5, .hdf5)
    - Pickle (.pkl, .pickle)
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        tuple: (pandas.DataFrame, error_message)
               Returns (df, None) on success
               Returns (None, error_str) on failure
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        loaders = {
            '.csv': lambda path: pd.read_csv(path, low_memory=False),
            '.tsv': lambda path: pd.read_csv(path, sep='\t', low_memory=False),
            '.txt': lambda path: pd.read_csv(path, low_memory=False),
            '.xlsx': lambda path: pd.read_excel(path),
            '.xls': lambda path: pd.read_excel(path),
            '.xlsm': lambda path: pd.read_excel(path),
            '.xlsb': lambda path: pd.read_excel(path),
            '.json': lambda path: pd.read_json(path),
            '.parquet': lambda path: pd.read_parquet(path),
            '.pq': lambda path: pd.read_parquet(path),
            '.feather': lambda path: pd.read_feather(path),
            '.sas7bdat': lambda path: pd.read_sas(path),
            '.xpt': lambda path: pd.read_sas(path, format='xport'),
            '.dta': lambda path: pd.read_stata(path),
            '.sav': lambda path: pd.read_spss(path),
            '.h5': lambda path: pd.read_hdf(path),
            '.hdf5': lambda path: pd.read_hdf(path),
            '.pkl': lambda path: pd.read_pickle(path),
            '.pickle': lambda path: pd.read_pickle(path),
        }
        
        if ext not in loaders:
            return None, f"Unsupported file format: {ext}\n\nSupported formats: CSV, Excel, JSON, Parquet, SAS, STATA, SPSS, and more."
        
        df = loaders[ext](file_path)
        
        if not isinstance(df, pd.DataFrame):
            return None, "Error: File loaded but did not produce a DataFrame."
        
        if df.empty:
            return None, "Error: File is empty (no data)."
        
        return df, None
        
    except FileNotFoundError:
        return None, f"Error: File not found.\n\n{file_path}"
    except pd.errors.EmptyDataError:
        return None, "Error: File is empty or has no data."
    except pd.errors.ParserError as e:
        return None, f"Error: Could not parse file.\n\n{str(e)}"
    except ValueError as e:
        return None, f"Error: Invalid data format.\n\n{str(e)}"
    except Exception as e:
        return None, f"Error loading file:\n\n{type(e).__name__}: {str(e)}"