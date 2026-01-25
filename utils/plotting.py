"""
Plotting utilities for binning visualization
Blue theme with darker Target color
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class WOEPlotter:
    """Create WOE plots for binned variables"""
    
    @staticmethod
    def create_woe_plot(binned_df, variable_name, figsize=(10, 6)):
        """
        Create WOE plot using blue theme with darker Target color
        
        Args:
            binned_df: DataFrame with binning results from binner class
            variable_name: Name of variable
            figsize: Figure size (width, height)
            
        Returns:
            Figure: Matplotlib figure
        """
        plot_df = binned_df.copy()
        
        # Extract metrics
        iv_value = round(plot_df['IV'].iloc[0], 4) if 'IV' in plot_df.columns else 0
        gini_value = round(plot_df['GINI'].iloc[0], 4) if 'GINI' in plot_df.columns else 0
        auc_value = round(plot_df['AUC'].iloc[0], 4) if 'AUC' in plot_df.columns else 0
        
        # Create figure
        fig = Figure(figsize=figsize, dpi=100)
        ax1 = fig.add_subplot(111)
        
        # Use Value column for x-axis labels
        x_labels = plot_df['Value'].astype(str).tolist()
        x_pos = np.arange(len(x_labels))
        
        # Color scheme - DARKER BLUE FOR TARGET
        target_color = '#1F618D'  # DARKER BLUE (as shown in your image)
        non_target_color = '#5DADE2'  # Light blue
        woe_color = '#2C3E50'  # Dark blue-gray
        
        # Stacked bars
        bar_width = 0.6
        ax1.bar(x_pos, plot_df['Target'], width=bar_width,
                color=target_color, label='Target (Bad)', alpha=0.9, 
                edgecolor='white', linewidth=0.5)
        ax1.bar(x_pos, plot_df['Non Target'], width=bar_width,
                bottom=plot_df['Target'], color=non_target_color, 
                label='Non-Target (Good)', alpha=0.85, 
                edgecolor='white', linewidth=0.5)
        
        ax1.set_xlabel('Bins', fontsize=11, fontweight='600', color='#2C3E50')
        ax1.set_ylabel('Observations', fontsize=11, fontweight='600', color='#2C3E50')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax1.tick_params(axis='both', labelsize=9, colors='#2C3E50')
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', color='#BDC3C7')
        ax1.set_facecolor('#F8F9FA')
        
        # WOE line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x_pos, plot_df['WOE'], marker='o', linestyle='-', 
                color=woe_color, linewidth=2.5, markersize=7, label='WOE',
                markerfacecolor=woe_color, markeredgecolor='white', 
                markeredgewidth=1.5)
        ax2.axhline(0, color='#7F8C8D', linestyle=':', linewidth=1.2, alpha=0.7)
        ax2.set_ylabel('Weight of Evidence (WOE)', fontsize=11, fontweight='600', color=woe_color)
        ax2.tick_params(axis='y', labelcolor=woe_color, labelsize=9)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Add WOE values on points
        for i, (x, y) in enumerate(zip(x_pos, plot_df['WOE'])):
            ax2.annotate(f'{y:.2f}', xy=(x, y), 
                        xytext=(0, 8 if y > 0 else -8), textcoords='offset points',
                        ha='center', va='bottom' if y > 0 else 'top',
                        fontsize=8, color=woe_color, fontweight='600',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=woe_color, alpha=0.7, linewidth=0.8))
        
        # Title
        ax1.set_title(f'{variable_name}', 
                     fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
        
        # Style spines
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for spine in ['left', 'bottom']:
            ax1.spines[spine].set_color('#BDC3C7')
            ax1.spines[spine].set_linewidth(1)
        ax2.spines['right'].set_color('#BDC3C7')
        ax2.spines['right'].set_linewidth(1)
        
        fig.tight_layout(pad=1.5)
        
        return fig