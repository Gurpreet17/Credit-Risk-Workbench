import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns

class CreditScoreModel:
    """
    Credit scoring model for binary classification with WOE-transformed features
    
    Parameters
    ----------
    classifier : object
        Sklearn-compatible classifier (e.g., LogisticRegression)
    
    score_params: dict
        Scoring parameters: {"pdo", "base_odds", "base_score", "population_odds"}
    
    binning_engine: object
        Binning object with binned_results attribute
    
    higher_score_better: bool (default=True)
        If True, higher scores indicate lower risk
    
    impute_missing: bool (default=True)
        Whether to impute missing value bins with mean points
    
    Attributes
    ----------
    scorecard_table: DataFrame
        Complete scorecard with points for each bin
    
    points_calculated: bool
        Whether points have been scaled
    
    analysis_data: list
        Stored data for calibration charts
    
    missing_fill_value: float
        Points assigned to missing values
    """
    
    def __init__(self, classifier, score_params, binning_engine, higher_score_better=True, impute_missing=False):
        self.classifier = classifier
        self.score_params = score_params
        self.binning_engine = binning_engine
        self.higher_score_better = higher_score_better
        self.impute_missing = impute_missing
        self.scorecard_table = None
        self.points_calculated = False
        self.analysis_data = None
        self.missing_fill_value = 0
        
    def train(self, X, y):
        """
        Train the scoring model
        
        Args:
            X: WOE-transformed feature matrix
            y: target vector
        """
        self.classifier.fit(X, y)
        
    def get_coefficients(self):
        """Extract model coefficients"""
        return self.classifier.coef_.flatten()
    
    def calculate_significance(self, X, y):
        """
        Calculate p-values for coefficients (memory-efficient version)
        
        Args:
            X: WOE-transformed feature matrix
            y: target vector
            
        Returns:
            list: tuples of (feature, p_value)
        """
        from scipy.stats import norm
        
        try:
            # For large datasets (>50k rows), use a subsample for efficiency
            if len(X) > 50000:
                print(f"[INFO] Using subsample of 50,000 rows for p-value calculation (original: {len(X)})")
                sample_idx = np.random.choice(len(X), 50000, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y[sample_idx]
            else:
                X_sample = X
                y_sample = y
            
            coeffs = self.classifier.coef_[0]
            probabilities = self.classifier.predict_proba(X_sample)
            
            # Clip probabilities to avoid numerical issues
            probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
            
            # Calculate weights as a vector (not diagonal matrix)
            weights = probabilities[:, 0] * probabilities[:, 1]
            
            # Weighted design matrix (element-wise)
            X_weighted = X_sample.values * np.sqrt(weights[:, np.newaxis])
            
            # Calculate X'WX using matrix multiplication (more efficient)
            XTX = np.dot(X_weighted.T, X_weighted)
            
            # Add ridge regularization to prevent singularity
            ridge_penalty = 1e-4
            XTX_regularized = XTX + ridge_penalty * np.eye(XTX.shape[0])
            
            try:
                # Try standard inversion
                covariance = np.linalg.inv(XTX_regularized)
            except np.linalg.LinAlgError:
                # If singular, use pseudoinverse
                print("[WARNING] Singular matrix detected. Using pseudoinverse for coefficient standard errors.")
                print("[INFO] This usually indicates multicollinearity. P-values may be unreliable.")
                covariance = np.linalg.pinv(XTX_regularized)
            
            # Calculate standard errors
            standard_errors = np.sqrt(np.abs(np.diag(covariance)))
            
            # Avoid division by zero
            standard_errors = np.where(standard_errors < 1e-10, 1e-10, standard_errors)
            
            # Calculate z-statistics and p-values
            z_statistics = coeffs / standard_errors
            p_values = 2 * (1 - norm.cdf(np.abs(z_statistics)))
            
            # Clip p-values to valid range [0, 1]
            p_values = np.clip(p_values, 0, 1)
            
            return list(zip(X.columns, p_values))
            
        except Exception as e:
            print(f"[ERROR] Could not calculate significance: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Return p_value = 1.0 (not significant) for all features as fallback
            print("[INFO] Returning p-value = 1.0 (not significant) for all variables")
            return [(col, 1.0) for col in X.columns]

    def build_scorecard_table(self, X):
        """
        Construct scorecard table from binning results
        
        Args:
            X: WOE-transformed feature matrix
        """
        features = X.columns
        coefficients = self.classifier.coef_.flatten()
        tables = []
        
        for idx, feature in enumerate(features):
            feature_table = self.binning_engine.binned_results[feature].copy()
            coef = coefficients[idx]
            feature_table.loc[:, "Coefficient"] = coef
            feature_table.loc[:, "Raw_Points"] = feature_table['WOE'] * coef
            feature_table.index.names = ['Bin_Index']
            feature_table.reset_index(level=0, inplace=True)
            tables.append(feature_table)
        
        self.scorecard_table = pd.concat(tables, ignore_index=True)
        
        # Add intercept as base score
        intercept = self.classifier.intercept_[0]
        
        base_row = pd.DataFrame([{
            'Bin_Index': -1,
            'Variable': 'BASE_SCORE',
            'Value': 'Intercept',
            'interval_end_exclude': np.nan,
            'interval_start_include': np.nan,
            'Total': np.nan,
            'Target Rate': np.nan,
            'Target': np.nan,
            'Non Target': np.nan,
            'Non Target %': np.nan,
            'WOE': 1.0,
            'bin_iv': np.nan,
            'feature_IV': np.nan,
            'AUC': np.nan,
            'Gini': np.nan,
            'Coefficient': intercept,
            'Raw_Points': intercept
        }])	
        self.scorecard_table = pd.concat([self.scorecard_table, base_row], ignore_index=True)
    
    def _calculate_points(self, raw_points):
        """
        Scale raw points to final scorecard points
        
        Args:
            raw_points: unscaled points (WOE * coefficient)
            
        Returns:
            scaled points
        """
        num_features = len(self.scorecard_table['Variable'].unique())
        direction = 1 if self.higher_score_better else -1
        
        pdo = self.score_params["pdo"]
        base_odds = self.score_params["base_odds"]
        base_score = self.score_params["base_score"]
        pop_odds = self.score_params["population_odds"]
        
        scaling_factor = pdo / np.log(2)
        intercept = self.classifier.intercept_[0]
        
        # Calculate base points including intercept
        offset = (((scaling_factor * np.log(pop_odds / base_odds)) + base_score) / num_features) + \
                 (direction * intercept * scaling_factor / num_features)
        
        # Calculate final points
        scaled_points = offset - (direction * raw_points * scaling_factor)
        
        return round(scaled_points)

    def finalize_points(self):
        """Scale all points in scorecard table"""
        if self.points_calculated:
            print("Points already calculated")
        else:
            self.scorecard_table.loc[:, "Points"] = self._calculate_points(self.scorecard_table["Raw_Points"])
            if self.impute_missing:
                # Calculate mean points for non-missing bins
                # Use the correct column name from your binning output
                non_missing_points = self.scorecard_table[
                    ~self.scorecard_table['interval_start_include'].isna()
                ]['Points']
                
                # Check if there are any non-missing bins
                if len(non_missing_points) > 0 and not non_missing_points.isna().all():
                    mean_points = round(non_missing_points.mean())
                    
                    # Assign to missing bins
                    self.scorecard_table.loc[
                        self.scorecard_table['interval_start_include'].isna(), "Points"
                    ] = mean_points
                    self.missing_fill_value = mean_points
                else:
                    # Fallback: no valid points to calculate mean
                    self.missing_fill_value = 0
                    print("Warning: Could not calculate mean points for missing bins")
                
            self.points_calculated = True
            
    def assign_points_to_data(self, raw_data, feature_list, target_column='bad'):
        """
        Assign scorecard points to each feature for each customer
        
        Args:
            raw_data: DataFrame with raw (non-WOE) feature values
            feature_list: list of features to score
            target_column: name of the target column in raw_data (default='bad')
            
        Returns:
            DataFrame with original features + point columns + score + target
        """
        result = raw_data[feature_list].copy()
        
        # Process each feature
        for feature in feature_list:
            # Get scorecard for this feature
            feature_scorecard = self.scorecard_table[
                self.scorecard_table['Variable'] == feature
            ].copy()
            
            
            # === DIAGNOSTIC ===
            print(f"\nProcessing feature: {feature}")
            print(f"  In numeric_cols: {feature in self.binning_engine.numeric_cols}")
            print(f"  In cat_cols: {feature in self.binning_engine.categoric_cols}")
            
            # Check what's in interval_start_include
            sample_interval = feature_scorecard['interval_start_include'].iloc[0]
            print(f"  Sample interval type: {type(sample_interval)}")
            print(f"  Sample interval value: {sample_interval}")
            
            if feature in self.binning_engine.numeric_cols:
                # === NUMERIC - SMART INTERVAL MATCHING ===
                bins_data = feature_scorecard[
                    ~feature_scorecard['interval_start_include'].isna()
                ].copy()
                
                # Initialize points column
                result[f'{feature}_pts'] = np.nan
                
                # Process each bin
                for idx, row in bins_data.iterrows():
                    lower = row['interval_start_include']
                    upper = row['interval_end_exclude']
                    points = row['Points']
                    
                    # Determine comparison logic based on actual values
                    if np.isinf(lower) and np.isinf(upper):
                        continue
                    elif lower == -np.inf:
                        mask = result[feature] < upper
                    elif upper == np.inf:
                        mask = result[feature] >= lower
                    elif lower == np.inf:
                        mask = result[feature] >= upper
                    elif upper == -np.inf:
                        mask = result[feature] < lower
                    else:
                        # Both finite - check which is actually lower
                        if lower < upper:
                            mask = (result[feature] >= lower) & (result[feature] < upper)
                        else:
                            mask = (result[feature] >= upper) & (result[feature] < lower)
                    
                    # Assign points
                    result.loc[mask, f'{feature}_pts'] = points
                
                # Handle missing
                missing_row = feature_scorecard[
                    feature_scorecard['interval_start_include'].isna()
                ]
                if not missing_row.empty:
                    result[f'{feature}_pts'] = result[f'{feature}_pts'].fillna(
                        missing_row['Points'].values[0]
                    )
            
            else:
                # === CATEGORIC - DIRECT MAPPING ===
                points_map = {}
                for _, row in feature_scorecard.iterrows():
                    category = row['interval_start_include']
                    pts = row['Points']
                    
                    if isinstance(category, list):
                        for cat in category:
                            points_map[str(cat)] = pts
                    elif pd.notna(category):
                        points_map[str(category)] = pts
                    else:
                        points_map['__MISSING__'] = pts
                
                # Map values
                mapped = result[feature].fillna('__MISSING__').astype(str).map(points_map)
                result[f'{feature}_pts'] = mapped
        
        # Get intercept points
        intercept_row = self.scorecard_table[
            self.scorecard_table['Variable'] == 'BASE_SCORE'
        ]
        intercept_points = intercept_row['Points'].values[0] if not intercept_row.empty else 0
        
        # Calculate total score
        point_columns = [col for col in result.columns if col.endswith('_pts')]
        result['score'] = result[point_columns].sum(axis=1) + intercept_points  # ← Changed to 'score'
        
        # Add target column at the end
        if target_column in raw_data.columns:
            result['target'] = raw_data[target_column].values  # ← Added target
        
        return result
    
    def plot_score_distribution(self, scored_data, save_path=None):
        """
        Plot kernel density distribution of scores
        
        Args:
            scored_data: DataFrame with 'score' and 'target' columns
            save_path: file path to save plot
        """
        plt.figure(figsize=(16, 10))

        # Plot KDE for non-defaults (target=0) - GREEN
        sns.kdeplot(
            data=scored_data[scored_data['target'] == 0]['score'],
            fill=True,
            color='green',
            alpha=0.5,
            label='Non-Default (Good Customers)',
            linewidth=2.5
        )

        # Plot KDE for defaults (target=1) - RED
        sns.kdeplot(
            data=scored_data[scored_data['target'] == 1]['score'],
            fill=True,
            color='red',
            alpha=0.5,
            label='Default (Bad Customers)',
            linewidth=2.5
        )

        # Styling
        plt.xlabel('Credit Score', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.title('Credit Score Distribution', fontsize=20, fontweight='bold')
        plt.legend(loc='upper left', fontsize=14, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_normalized_distribution(self, scored_data, save_path=None):
        """
        Plot normalized score distributions (single plot)
        
        Args:
            scored_data: DataFrame with 'score' and 'target' columns
            save_path: file path to save plot
        """
        plt.figure(figsize=(16, 10))
        
        non_default = scored_data[scored_data['target'] == 0]['score']
        default = scored_data[scored_data['target'] == 1]['score']
        
        n_non_default = len(non_default)
        n_default = len(default)
        
        # Plot normalized histograms with KDE overlay
        plt.hist(
            non_default,
            bins=50,
            weights=np.ones(n_non_default) / n_non_default,
            alpha=0.3,
            color='green',
            label='Non-Default (Good)',
            edgecolor='darkgreen',
            linewidth=1.2
        )
        
        plt.hist(
            default,
            bins=50,
            weights=np.ones(n_default) / n_default,
            alpha=0.3,
            color='red',
            label='Default (Bad)',
            edgecolor='darkred',
            linewidth=1.2
        )
        
        # Overlay KDE
        sns.kdeplot(
            data=non_default,
            color='darkgreen',
            linewidth=2.5,
            label='Non-Default KDE'
        )
        
        sns.kdeplot(
            data=default,
            color='darkred',
            linewidth=2.5,
            label='Default KDE'
        )

        plt.xlabel('Credit Score', fontsize=16)
        plt.ylabel('Normalized Frequency', fontsize=16)
        plt.title('Normalized Credit Score Distribution', fontsize=20, fontweight='bold')
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    def plot_roc(self, scored_data, save_path=None):
        """
        Plot ROC curve
        
        Args:
            scored_data: DataFrame with 'score' and 'target' columns
                        OR dict with dataset names as keys and DataFrames as values
            save_path: file path to save plot
        """
        plt.figure(figsize=(16, 8))
        plt.suptitle('ROC Curve', fontsize=20, fontweight='bold')

        # Handle both single DataFrame and dict of DataFrames
        if isinstance(scored_data, dict):
            datasets = scored_data
        else:
            datasets = {'Data': scored_data}

        for name, data in datasets.items():
            # Negate scores since higher score = lower risk
            predictions = -data['score']
            
            false_pos_rate, true_pos_rate, _ = metrics.roc_curve(data['target'], predictions)
            area_under_curve = metrics.auc(false_pos_rate, true_pos_rate)

            print(f"{name} AUC: {area_under_curve:.4f}")
            plt.plot(false_pos_rate, true_pos_rate, label=f'{name} AUC = {area_under_curve:.4f}', linewidth=2.5)

        plt.grid(visible=True, which='both', color='0.65', linestyle='--', alpha=0.3)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Classifier', linewidth=2)
        plt.legend(loc='lower right', fontsize=14, framealpha=0.9)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_calibration_chart(self, test_data, train_data, num_quantiles=30, save_path=None):
        """
        Plot observed vs expected odds
        
        Args:
            test_data: DataFrame with 'target' and 'score' columns
            train_data: DataFrame with 'target' and 'score' columns
            num_quantiles: number of score quantiles
            save_path: file path to save plot
        """
        from copy import deepcopy
        
        pdo = self.score_params["pdo"]
        base_odds = self.score_params["base_odds"]
        base_score = self.score_params["base_score"]
        
        test_copy = deepcopy(test_data)
        train_copy = deepcopy(train_data)

        datasets = [test_copy, train_copy]
        print(f"Dataset sizes - Test: {len(test_copy)}, Train: {len(train_copy)}")

        analysis_results = []

        for dataset in datasets:
            dataset['score_quantile'], _ = pd.qcut(
                dataset['score'], q=num_quantiles, labels=False, retbins=True, duplicates='drop'
            )

            aggregated = dataset.groupby(['score_quantile'])[
                ['score_quantile', 'target', 'score']
            ].agg(['mean', 'count', 'sum'])
            aggregated.columns = ['_'.join(col) for col in aggregated.columns]
            aggregated.reset_index()

            aggregated['observed_odds'] = (1 - aggregated['target_mean'])/aggregated['target_mean']
            aggregated['log_observed_odds'] = np.log(aggregated['observed_odds'])
            aggregated['log_observed_odds'] = aggregated['log_observed_odds'].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            aggregated['expected_odds'] = base_odds * 2 ** ((aggregated['score_mean'] - base_score) / pdo)
            aggregated['log_expected_odds'] = np.log(aggregated['expected_odds'])
            
            analysis_results.append(aggregated)

        self.analysis_data = analysis_results

        # Plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(14, 11))
        
        for idx, agg_data in enumerate(self.analysis_data):
            if idx == 0:  # Test set
                ax.scatter(x='score_mean', y='log_expected_odds', data=agg_data, 
                          color='black', label='Expected Odds')
                ax.scatter(x='score_mean', y='log_observed_odds', data=agg_data, 
                          label='Observed Odds (Test)', alpha=0.6)
            else:  # Train set
                ax.scatter(x='score_mean', y='log_observed_odds', data=agg_data, 
                          label='Observed Odds (Train)', alpha=0.6)

        plt.xlabel('Credit Score')
        plt.ylabel('log(Odds)')
        plt.title('Calibration Chart')
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_calibration_with_ci(self, save_path=None):
        """
        Plot calibration with confidence intervals
        
        Args:
            save_path: file path to save plot
        """
        from scipy.stats import t, linregress
        
        if self.analysis_data is None:
            print("Error: Run plot_calibration_chart() first to generate analysis data")
            return
        
        pdo = self.score_params["pdo"]
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(14, 11))
        plt.title('Calibration Chart with Confidence Intervals')

        test_agg = self.analysis_data[0]
        
        sns.regplot(x='score_mean', y='log_observed_odds', ax=ax, data=test_agg, 
                   ci=95, label='Test Observed')
        sns.regplot(x='score_mean', y='log_expected_odds', ax=ax, data=test_agg, 
                   ci=None, color='black', label='Expected', 
                   line_kws={"linestyle": '--', "alpha": 1})
        sns.regplot(x='score_mean', y='log_observed_odds', ax=ax, data=self.analysis_data[1], 
                   ci=95, label='Train Observed')

        ax.set(xlabel='Credit Score', ylabel='log(Odds)')
        ax.legend(loc="upper left")

        # Calculate fit parameters
        fit_result = linregress(test_agg['score_mean'], test_agg['log_observed_odds'])
        t_value = abs(t.ppf(0.025, len(test_agg) - 2))
        
        print(f"Slope: {fit_result.slope:.4f}")
        print(f"Intercept: {fit_result.intercept:.4f}")
        print(f"R-squared: {fit_result.rvalue**2:.4f}")

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

        print(f"\nObserved PDO: {fit_result.slope:.2f} ± {t_value * fit_result.stderr:.2f}")
        print(f"Expected PDO: {pdo:.2f}")
        print(f"Within 95% CI: {abs(fit_result.slope - pdo) < (t_value * fit_result.stderr)}")

    def analyze_coefficient_correlation(self, X, threshold=0.4):
        """
        Analyze correlation between WOE features (simplified)
        
        Args:
            X: WOE-transformed feature matrix
            threshold: correlation threshold for flagging
            
        Returns:
            tuple: (high_correlation_pairs, full_correlation_matrix)
        """
        try:
            # Simple correlation matrix of WOE features
            corr_df = X.corr()
            
            # Find high correlations
            corr_long = corr_df.stack().reset_index()
            corr_long.columns = ['Feature1', 'Feature2', 'Correlation']
            
            high_corr = corr_long[
                (corr_long['Feature1'] != corr_long['Feature2']) &
                (corr_long['Correlation'].abs() > threshold)
            ]
            
            if len(high_corr) > 0:
                high_corr['Pair'] = high_corr.apply(
                    lambda row: tuple(sorted([row['Feature1'], row['Feature2']])), axis=1
                )
                unique_high_corr = high_corr.drop_duplicates(subset='Pair').drop(columns='Pair')
                
                print(f"\n[CORRELATION] Feature pairs with |correlation| > {threshold}:")
                print(unique_high_corr.to_string(index=False))
            else:
                unique_high_corr = pd.DataFrame()
            
            return unique_high_corr, corr_df
            
        except Exception as e:
            print(f"[ERROR] Could not calculate correlation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Return empty DataFrames as fallback
            empty_df = pd.DataFrame()
            empty_corr = pd.DataFrame(
                np.eye(len(X.columns)),
                columns=list(X.columns),
                index=list(X.columns)
            )
            return empty_df, empty_corr
    
    
    def calculate_ks_statistic(self, scored_data):
        """
        Calculate Kolmogorov-Smirnov statistic
        
        Args:
            scored_data: DataFrame with 'score' and 'target' columns
            
        Returns:
            tuple: (ks_stat, ks_score, ks_data)
                - ks_stat: KS statistic value
                - ks_score: Score where KS is maximum
                - ks_data: DataFrame with cumulative distributions
        """
        # Separate good and bad customers
        good_scores = scored_data[scored_data['target'] == 0]['score'].values
        bad_scores = scored_data[scored_data['target'] == 1]['score'].values
        
        # Get all unique scores (sorted)
        all_scores = np.sort(np.unique(np.concatenate([good_scores, bad_scores])))
        
        # Calculate cumulative distributions
        ks_data = []
        max_ks = 0
        max_ks_score = None
        
        for score in all_scores:
            # Cumulative % of good customers with score <= threshold
            cum_good_pct = np.sum(good_scores <= score) / len(good_scores)
            
            # Cumulative % of bad customers with score <= threshold
            cum_bad_pct = np.sum(bad_scores <= score) / len(bad_scores)
            
            # KS at this score
            ks = abs(cum_good_pct - cum_bad_pct)
            
            ks_data.append({
                'score': score,
                'cum_good_pct': cum_good_pct,
                'cum_bad_pct': cum_bad_pct,
                'ks': ks
            })
            
            # Track maximum KS
            if ks > max_ks:
                max_ks = ks
                max_ks_score = score
        
        ks_df = pd.DataFrame(ks_data)
        
        return max_ks, max_ks_score, ks_df


    def plot_ks_chart(self, scored_data, save_path=None):
        """
        Plot KS chart showing cumulative distributions
        
        Args:
            scored_data: DataFrame with 'score' and 'target' columns
            save_path: file path to save plot
        """
        # Calculate KS
        ks_stat, ks_score, ks_data = self.calculate_ks_statistic(scored_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot cumulative distributions
        ax.plot(ks_data['score'], ks_data['cum_good_pct'], 
                color='green', linewidth=2.5, label='Good Customers (Cumulative %)')
        ax.plot(ks_data['score'], ks_data['cum_bad_pct'], 
                color='red', linewidth=2.5, label='Bad Customers (Cumulative %)')
        
        # Plot KS line at maximum separation
        ks_row = ks_data[ks_data['score'] == ks_score].iloc[0]
        ax.vlines(x=ks_score, 
                ymin=ks_row['cum_bad_pct'], 
                ymax=ks_row['cum_good_pct'],
                colors='blue', linewidth=3, linestyles='dashed',
                label=f'KS = {ks_stat:.3f} at Score = {ks_score:.0f}')
        
        # Add annotation
        ax.annotate(f'KS = {ks_stat:.1%}\nat Score = {ks_score:.0f}',
                xy=(ks_score, (ks_row['cum_good_pct'] + ks_row['cum_bad_pct']) / 2),
                xytext=(ks_score + 20, (ks_row['cum_good_pct'] + ks_row['cum_bad_pct']) / 2),
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))
        
        # Styling
        ax.set_xlabel('Credit Score', fontsize=16)
        ax.set_ylabel('Cumulative %', fontsize=16)
        ax.set_title('Kolmogorov-Smirnov (KS) Chart', fontsize=20, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return ks_stat, ks_score, ks_data