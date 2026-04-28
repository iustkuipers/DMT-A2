"""
EDA Module for Expedia Hotel Ranking (Assignment 2)
Modular exploratory data analysis without feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HotelEDA:
    """
    Exploratory Data Analysis class for Expedia hotel ranking data.
    Handles both training and test sets.
    """
    
    def __init__(self, train_path, test_path=None):
        """
        Initialize with data paths.
        
        Args:
            train_path: path to training set
            test_path: path to test set (optional)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.df_train = None
        self.df_test = None
        self.numeric_cols = None
        self.categorical_cols = None
        
    def load_data(self):
        """Load training and test data."""
        print("Loading training data...")
        self.df_train = pd.read_csv(self.train_path)
        print(f"Training set shape: {self.df_train.shape}")
        
        if self.test_path:
            print("Loading test data...")
            self.df_test = pd.read_csv(self.test_path)
            print(f"Test set shape: {self.df_test.shape}")
        
        # Identify column types
        self.numeric_cols = self.df_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df_train.select_dtypes(include=['object']).columns.tolist()
        
        return self.df_train, self.df_test
    
    # ============ PHASE 1: Data Quality & Logical Structure ============
    
    def basic_counts(self):
        """Basic counts: unique values and average hotels per search."""
        print("\n" + "="*60)
        print("PHASE 1: DATA QUALITY & LOGICAL STRUCTURE")
        print("="*60)
        print("\nBASIC COUNTS:")
        print(f"  Unique srch_ids: {self.df_train['srch_id'].nunique()}")
        print(f"  Unique prop_ids: {self.df_train['prop_id'].nunique()}")
        
        avg_hotels_per_search = self.df_train.groupby('srch_id').size().mean()
        print(f"  Average hotels per search: {avg_hotels_per_search:.2f}")
        
        if self.df_test is not None:
            print(f"\n  Test set - Unique srch_ids: {self.df_test['srch_id'].nunique()}")
            print(f"  Test set - Unique prop_ids: {self.df_test['prop_id'].nunique()}")
            
            # Check overlap
            train_srch = set(self.df_train['srch_id'].unique())
            test_srch = set(self.df_test['srch_id'].unique())
            overlap = len(train_srch & test_srch)
            overlap_pct = 100 * overlap / len(test_srch)
            print(f"  Train-test srch_id overlap: {overlap} ({overlap_pct:.1f}% of test)")
            print(f"  Note: Overlap is OK if same user searching at different times")
    
    def position_availability(self):
        """Check position field availability (only in train)."""
        print("\nPOSITION FIELD:")
        if 'position' in self.df_train.columns:
            print(f"  ✓ 'position' exists in training set (n={len(self.df_train)})")
            print(f"  Position range: {self.df_train['position'].min()} to {self.df_train['position'].max()}")
        else:
            print("  ✗ 'position' not found in training set")
        
        if self.df_test is not None and 'position' in self.df_test.columns:
            print(f"  ✓ 'position' exists in test set")
        else:
            print("  ✗ 'position' not in test set (as expected)")
    
    def range_and_type_checks(self):
        """Check ranges and special values in key fields."""
        print("\nRANGE & TYPE CHECKS:")
        
        if 'prop_starrating' in self.df_train.columns:
            zeros = (self.df_train['prop_starrating'] == 0).sum()
            print(f"  prop_starrating = 0: {zeros} ({100*zeros/len(self.df_train):.2f}%) [unknown, not bad]")
        
        if 'prop_review_score' in self.df_train.columns:
            zeros = (self.df_train['prop_review_score'] == 0).sum()
            nulls = self.df_train['prop_review_score'].isna().sum()
            print(f"  prop_review_score = 0: {zeros} (no reviews)")
            print(f"  prop_review_score = null: {nulls} (unknown)")
        
        if 'price_usd' in self.df_train.columns:
            print(f"  price_usd range: ${self.df_train['price_usd'].min():.2f} to ${self.df_train['price_usd'].max():.2f}")
            outliers = (self.df_train['price_usd'] > 10000).sum()
            print(f"  Outliers (>$10,000): {outliers}")
        
        if 'prop_log_historical_price' in self.df_train.columns:
            zeros = (self.df_train['prop_log_historical_price'] == 0).sum()
            print(f"  prop_log_historical_price = 0: {zeros} (not sold last period)")
    
    def competitor_missingness(self):
        """Quantify competitor field missingness."""
        print("\nCOMPETITOR FIELDS:")
        comp_cols = [col for col in self.df_train.columns if col.startswith('comp')]
        if comp_cols:
            for col in comp_cols[:3]:  # Show first 3
                missing_pct = 100 * self.df_train[col].isna().sum() / len(self.df_train)
                print(f"  {col}: {missing_pct:.1f}% missing")
            if len(comp_cols) > 3:
                print(f"  ... and {len(comp_cols) - 3} more competitor columns")
        else:
            print("  No competitor columns found")
    
    def missingness_overview(self):
        """Bar chart of % missing per column."""
        print("\nMISSINGNESS OVERVIEW:")
        
        missing_pct = 100 * self.df_train.isna().sum() / len(self.df_train)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) > 0:
            print(f"  Total columns with missing data: {len(missing_pct)}")
            print(f"  Top 5 columns by missingness:")
            for col, pct in missing_pct.head(5).items():
                print(f"    {col}: {pct:.1f}%")
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_pct.head(15).plot(kind='barh', ax=ax, color='salmon')
            ax.set_xlabel('% Missing')
            ax.set_title('Data Quality: Missing Values by Column')
            plt.tight_layout()
            plt.savefig('eda_01_missingness.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_01_missingness.png")
            plt.close()
        else:
            print("  No missing values found!")
    
    # ============ PHASE 2: Data Exploration ============
    
    def target_variable_analysis(self):
        """Analyze click and booking distributions."""
        print("\n" + "="*60)
        print("PHASE 2: DATA EXPLORATION")
        print("="*60)
        print("\nTARGET VARIABLE ANALYSIS:")
        
        if 'click_bool' in self.df_train.columns:
            click_rate = self.df_train['click_bool'].mean()
            print(f"  Click rate: {100*click_rate:.2f}%")
        
        if 'booking_bool' in self.df_train.columns:
            booking_rate = self.df_train['booking_bool'].mean()
            print(f"  Booking rate: {100*booking_rate:.2f}% (extreme imbalance)")
            
            # Bookings per search
            bookings_per_search = self.df_train.groupby('srch_id')['booking_bool'].sum()
            print(f"  Max bookings per search: {bookings_per_search.max()}")
            print(f"  Mean bookings per search: {bookings_per_search.mean():.3f}")
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            targets = ['click_bool', 'booking_bool']
            for ax, target in zip(axes, targets):
                if target in self.df_train.columns:
                    counts = self.df_train[target].value_counts()
                    counts.plot(kind='bar', ax=ax, color=['gray', 'green'])
                    ax.set_title(f'{target.replace("_", " ").title()} Distribution')
                    ax.set_ylabel('Count')
                    ax.set_xticklabels(['No', 'Yes'], rotation=0)
                    rates = 100 * self.df_train[target].value_counts(normalize=True)
                    for i, (idx, v) in enumerate(rates.items()):
                        ax.text(i, counts.iloc[i], f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('eda_02_target_distribution.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_02_target_distribution.png")
            plt.close()
    
    def position_bias_analysis(self):
        """Analyze position bias (click/booking by position, random vs non-random)."""
        print("\nPOSITION BIAS ANALYSIS:")
        
        if 'position' not in self.df_train.columns:
            print("  Skipping: 'position' column not found")
            return
        
        if 'random_bool' not in self.df_train.columns:
            print("  Skipping: 'random_bool' column not found")
            return
        
        # Split by random_bool
        random_results = self.df_train[self.df_train['random_bool'] == 1]
        non_random_results = self.df_train[self.df_train['random_bool'] == 0]
        
        print(f"  Random results: {len(random_results)} ({100*len(random_results)/len(self.df_train):.1f}%)")
        print(f"  Non-random results: {len(non_random_results)} ({100*len(non_random_results)/len(self.df_train):.1f}%)")
        
        # Position analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (data, title) in enumerate([
            (non_random_results, 'Non-Random (Expedia Algorithm)'),
            (random_results, 'Random Ranking')
        ]):
            position_stats = data.groupby('position').agg({
                'click_bool': 'mean' if 'click_bool' in data.columns else lambda x: 0,
                'booking_bool': 'mean' if 'booking_bool' in data.columns else lambda x: 0
            }).reset_index()
            
            if 'click_bool' in data.columns and 'booking_bool' in data.columns:
                ax = axes[idx]
                ax.plot(position_stats['position'], position_stats['click_bool']*100, 
                       marker='o', label='Click Rate', linewidth=2)
                ax.plot(position_stats['position'], position_stats['booking_bool']*100,
                       marker='s', label='Booking Rate', linewidth=2)
                ax.set_xlabel('Position')
                ax.set_ylabel('Rate (%)')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eda_03_position_bias.png', dpi=100, bbox_inches='tight')
        print("  Saved: eda_03_position_bias.png")
        plt.close()
    
    def property_level_features(self):
        """Analyze property-level features."""
        print("\nPROPERTY-LEVEL FEATURES:")
        
        if 'prop_starrating' in self.df_train.columns:
            unknown_pct = 100 * (self.df_train['prop_starrating'] == 0).sum() / len(self.df_train)
            print(f"  prop_starrating - Unknown (0-star): {unknown_pct:.1f}%")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            # Include 0-star (unknown) values
            star_dist = self.df_train['prop_starrating'].value_counts().sort_index()
            star_dist.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title('Hotel Star Rating Distribution (Including 0=Unknown)')
            ax.set_xlabel('Star Rating')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            plt.savefig('eda_04_starrating.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_04_starrating.png")
            plt.close()
        
        if 'prop_review_score' in self.df_train.columns:
            no_reviews = (self.df_train['prop_review_score'] == 0).sum()
            print(f"  prop_review_score - No reviews: {100*no_reviews/len(self.df_train):.1f}%")
        
        if 'promotion_flag' in self.df_train.columns:
            promo_pct = 100 * self.df_train['promotion_flag'].mean()
            print(f"  Promoted hotels: {promo_pct:.1f}%")
    
    def price_analysis(self):
        """Analyze price features."""
        print("\nPRICE ANALYSIS:")
        
        if 'price_usd' in self.df_train.columns:
            price = self.df_train['price_usd']
            print(f"  price_usd - Mean: ${price.mean():.2f}, Median: ${price.median():.2f}")
            print(f"  price_usd - Std: ${price.std():.2f}, Max: ${price.max():.2f}")
            
            # Log-scale visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(price[price > 0], bins=50, color='skyblue', edgecolor='black')
            axes[0].set_xlabel('Price (USD)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Price Distribution')
            
            axes[1].hist(np.log1p(price[price > 0]), bins=50, color='lightgreen', edgecolor='black')
            axes[1].set_xlabel('Log(Price + 1)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Price Distribution (Log Scale)')
            
            plt.tight_layout()
            plt.savefig('eda_05_price_distribution.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_05_price_distribution.png")
            plt.close()
    
    def search_level_features(self):
        """Analyze search-level features."""
        print("\nSEARCH-LEVEL FEATURES:")
        
        search_cols = ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_room_count']
        for col in search_cols:
            if col in self.df_train.columns:
                print(f"  {col}: mean={self.df_train[col].mean():.2f}, median={self.df_train[col].median():.1f}")
        
        if 'srch_saturday_night_bool' in self.df_train.columns:
            weekend_pct = 100 * self.df_train['srch_saturday_night_bool'].mean()
            print(f"  Saturday night stay: {weekend_pct:.1f}%")
    
    # ============ PHASE 3: Importance & Dependencies ============
    
    def correlation_analysis(self):
        """Point-biserial correlation with target variables."""
        print("\n" + "="*60)
        print("PHASE 3: IMPORTANCE & DEPENDENCIES")
        print("="*60)
        print("\nCORRELATION WITH TARGET VARIABLES:")
        
        targets = ['booking_bool', 'click_bool']
        
        for target in targets:
            if target not in self.df_train.columns:
                continue
            
            print(f"\n  {target}:")
            correlations = self.df_train[self.numeric_cols].corrwith(self.df_train[target]).abs().sort_values(ascending=False)
            correlations = correlations[correlations > 0.01]  # Only significant correlations
            
            print(f"  Top 10 correlated features:")
            for col, corr in correlations.head(10).items():
                print(f"    {col}: {corr:.4f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            correlations.head(15).sort_values().plot(kind='barh', ax=ax, color='teal')
            ax.set_xlabel('Absolute Correlation')
            ax.set_title(f'Feature Correlation with {target.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(f'eda_06_{target}_correlation.png', dpi=100, bbox_inches='tight')
            print(f"  Saved: eda_06_{target}_correlation.png")
            plt.close()
    
    def propid_aggregates(self):
        """
        Per-property aggregates: mean booking/click rate, mean/std/median of numeric features.
        This is the Kaggle winning insight!
        """
        print("\nPROP_ID AGGREGATES (Winning Features):")
        print("  Computing per-property statistics...")
        
        # Target variable aggregates
        prop_agg = self.df_train.groupby('prop_id').agg({
            'booking_bool': ['mean', 'sum', 'count'] if 'booking_bool' in self.df_train.columns else lambda x: 0,
            'click_bool': ['mean', 'sum'] if 'click_bool' in self.df_train.columns else lambda x: 0,
        }).reset_index()
        
        if 'booking_bool' in self.df_train.columns:
            print(f"  Properties: {len(prop_agg)}")
            print(f"  Booking rate by property - Min: {prop_agg[('booking_bool', 'mean')].min():.4f}, "
                  f"Max: {prop_agg[('booking_bool', 'mean')].max():.4f}")
            
            # Numeric feature aggregates per prop_id
            numeric_aggs = {}
            for col in self.numeric_cols:
                if col not in ['booking_bool', 'click_bool', 'srch_id', 'prop_id', 'position']:
                    agg_mean = self.df_train.groupby('prop_id')[col].mean()
                    agg_std = self.df_train.groupby('prop_id')[col].std()
                    agg_median = self.df_train.groupby('prop_id')[col].median()
                    
                    numeric_aggs[f'{col}_mean'] = agg_mean
                    numeric_aggs[f'{col}_std'] = agg_std
                    numeric_aggs[f'{col}_median'] = agg_median
            
            print(f"  Numeric features aggregated: {len([k for k in numeric_aggs.keys() if '_mean' in k])} columns")
            
            # Save aggregates to CSV for reference
            prop_agg_numeric = pd.DataFrame(numeric_aggs)
            prop_agg_numeric.to_csv('prop_id_aggregates.csv')
            print("  Saved: prop_id_aggregates.csv")
            
            # Visualization: variance in booking rate
            fig, ax = plt.subplots(figsize=(10, 5))
            prop_agg[('booking_bool', 'mean')].hist(bins=50, ax=ax, color='coral', edgecolor='black')
            ax.set_xlabel('Property Booking Rate')
            ax.set_ylabel('Number of Properties')
            ax.set_title('Distribution of Booking Rates Across Properties (Key Predictive Feature)')
            plt.tight_layout()
            plt.savefig('eda_07_propid_booking_rate.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_07_propid_booking_rate.png")
            plt.close()
    
    def within_query_features(self):
        """Within-query relative features (rank within search)."""
        print("\nWITHIN-QUERY RELATIVE FEATURES:")
        
        if 'price_usd' in self.df_train.columns:
            # Price rank within each search
            self.df_train['price_rank'] = self.df_train.groupby('srch_id')['price_usd'].rank()
            
            if 'booking_bool' in self.df_train.columns:
                price_rank_agg = self.df_train.groupby('price_rank')['booking_bool'].mean()
                print(f"  Price rank impact: cheapest hotels have booking rate "
                      f"{100*price_rank_agg.iloc[0]:.2f}%")
        
        if 'prop_starrating' in self.df_train.columns:
            # Starrating rank within search
            self.df_train['starrating_rank'] = self.df_train.groupby('srch_id')['prop_starrating'].rank(ascending=False)
            print(f"  Star rating ranks computed")
    
    # ============ Additional EDA for LambdaMART ============
    
    def srch_id_overlap_analysis(self):
        """Analyze train-test srch_id overlap for data leak detection."""
        print("\nTRAIN-TEST OVERLAP ANALYSIS (Data Leak Check):")
        
        if self.df_test is None:
            print("  Skipping: no test set")
            return
        
        train_srch = set(self.df_train['srch_id'].unique())
        test_srch = set(self.df_test['srch_id'].unique())
        overlap_srch_ids = train_srch & test_srch
        
        if len(overlap_srch_ids) == 0:
            print("  ✓ No overlap (good for time-series safety)")
        else:
            print(f"  Overlapping srch_ids: {len(overlap_srch_ids)}")
            
            # Check if same datetime
            if 'datetime' in self.df_train.columns or 'date_time' in self.df_train.columns:
                datetime_col = 'datetime' if 'datetime' in self.df_train.columns else 'date_time'
                
                overlap_train = self.df_train[self.df_train['srch_id'].isin(overlap_srch_ids)]
                overlap_test = self.df_test[self.df_test['srch_id'].isin(overlap_srch_ids)]
                
                # Check for exact datetime matches
                overlap_by_srch = overlap_train.groupby('srch_id')[datetime_col].first()
                test_by_srch = overlap_test.groupby('srch_id')[datetime_col].first()
                
                same_datetime = sum(overlap_by_srch == test_by_srch)
                print(f"  Same searches with identical datetime: {same_datetime}")
                print(f"  Different searches (repeat user): {len(overlap_srch_ids) - same_datetime}")
            else:
                print("  No datetime column found — cannot check for data leak")
                print("  But overlap with different srch_ids suggests repeat users (likely OK)")
    
    def propid_appearance_distribution(self):
        """Distribution of how many times each property appears (shrinkage analysis)."""
        print("\nPROP_ID APPEARANCE DISTRIBUTION (for Shrinkage):")
        
        prop_counts = self.df_train['prop_id'].value_counts()
        
        print(f"  Properties appearing 1x: {(prop_counts == 1).sum()} ({100*(prop_counts == 1).sum()/len(prop_counts):.1f}%)")
        print(f"  Properties appearing 2-5x: {((prop_counts >= 2) & (prop_counts <= 5)).sum()} ({100*((prop_counts >= 2) & (prop_counts <= 5)).sum()/len(prop_counts):.1f}%)")
        print(f"  Properties appearing <5x total: {(prop_counts < 5).sum()} ({100*(prop_counts < 5).sum()/len(prop_counts):.1f}%)")
        print(f"  Properties appearing 6-10x: {((prop_counts >= 6) & (prop_counts <= 10)).sum()}")
        print(f"  Properties appearing >10x: {(prop_counts > 10).sum()}")
        print(f"  Max appearances: {prop_counts.max()}")
        print(f"  Mean appearances: {prop_counts.mean():.1f}")
        print(f"  ⚠ Properties with <5 appearances are high-variance estimates — apply shrinkage")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(prop_counts, bins=50, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Number of Appearances per Property')
        axes[0].set_ylabel('Count of Properties')
        axes[0].set_title('Property Appearance Distribution (Unfiltered)')
        axes[0].set_yscale('log')
        
        # Log-scale histogram
        prop_counts_filtered = prop_counts[prop_counts > 0]
        axes[1].hist(prop_counts_filtered, bins=50, color='coral', edgecolor='black')
        axes[1].set_xlabel('Number of Appearances per Property')
        axes[1].set_ylabel('Count of Properties')
        axes[1].set_title('Property Appearance Distribution (Showing Most Properties)')
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('eda_08_propid_appearance.png', dpi=100, bbox_inches='tight')
        print("  Saved: eda_08_propid_appearance.png")
        plt.close()
    
    def within_query_relative_plots(self):
        """Binned plots: price rank and star rank vs booking rate (LambdaMART-specific)."""
        print("\nWITHIN-QUERY RELATIVE FEATURES (LambdaMART-critical):")
        
        if 'price_rank' not in self.df_train.columns:
            self.df_train['price_rank'] = self.df_train.groupby('srch_id')['price_usd'].rank()
        if 'starrating_rank' not in self.df_train.columns:
            self.df_train['starrating_rank'] = self.df_train.groupby('srch_id')['prop_starrating'].rank(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Price rank vs booking (full curve)
        if 'booking_bool' in self.df_train.columns:
            # Show wider range for price rank
            max_price_rank = min(25, int(self.df_train.groupby('srch_id').size().max()))
            price_rank_stats = self.df_train[self.df_train['price_rank'] <= max_price_rank].groupby('price_rank')['booking_bool'].agg(['mean', 'count'])
            
            axes[0].plot(price_rank_stats.index, price_rank_stats['mean']*100, marker='o', linewidth=2, markersize=6, color='steelblue')
            axes[0].fill_between(price_rank_stats.index, price_rank_stats['mean']*100, alpha=0.3, color='steelblue')
            axes[0].set_xlabel('Price Rank Within Search (1=Cheapest)')
            axes[0].set_ylabel('Booking Rate (%)')
            axes[0].set_title('Price Rank Impact on Bookings (Full Curve)')
            axes[0].grid(True, alpha=0.3)
            
            cheapest_rate = price_rank_stats.loc[1, 'mean'] * 100
            most_expensive = price_rank_stats.index.max()
            expensive_rate = price_rank_stats.loc[most_expensive, 'mean'] * 100
            print(f"  Price rank 1 (cheapest) booking rate: {cheapest_rate:.2f}%")
            print(f"  Price rank {int(most_expensive)} booking rate: {expensive_rate:.2f}%")
            print(f"  Decay: {cheapest_rate - expensive_rate:.2f}pp (strong within-query price effect)")
        
        # Star rank vs booking
        if 'booking_bool' in self.df_train.columns:
            star_rank_stats = self.df_train[self.df_train['starrating_rank'] <= 10].groupby('starrating_rank')['booking_bool'].agg(['mean', 'count'])
            
            axes[1].bar(star_rank_stats.index, star_rank_stats['mean']*100, color='lightcoral', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Star Rating Rank Within Search (1=Highest)')
            axes[1].set_ylabel('Booking Rate (%)')
            axes[1].set_title('Star Rank Impact on Bookings (Top 10)')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add counts
            for i, (idx, row) in enumerate(star_rank_stats.iterrows()):
                axes[1].text(idx, row['mean']*100 + 0.1, f"n={int(row['count'])}", ha='center', fontsize=8)
            
            print(f"  Highest star (rank 1) booking rate: {star_rank_stats.loc[1, 'mean']*100:.2f}%")
        
        plt.tight_layout()
        plt.savefig('eda_09_within_query_ranks.png', dpi=100, bbox_inches='tight')
        print("  Saved: eda_09_within_query_ranks.png")
        plt.close()
    
    def competitor_signal_analysis(self):
        """Booking rate by competitor rate values (despite high missingness)."""
        print("\nCOMPETITOR SIGNAL ANALYSIS:")
        
        comp_cols = [col for col in self.df_train.columns if col.startswith('comp') and col.endswith('_rate')]
        
        if not comp_cols or 'booking_bool' not in self.df_train.columns:
            print("  Skipping: no competitor columns or booking_bool")
            return
        
        # Focus on comp1_rate (primary competitor)
        if 'comp1_rate' in self.df_train.columns:
            print("  comp1_rate analysis (Expedia vs. primary competitor):")
            
            comp_stats = self.df_train.groupby('comp1_rate', observed=True)['booking_bool'].agg(['mean', 'count'])
            
            for idx, row in comp_stats.iterrows():
                if pd.isna(idx):
                    print(f"    Missing: {100*row['mean']:.2f}% booking (n={int(row['count'])})")
                else:
                    meaning = {-1: "Expedia cheaper", 0: "Same price", 1: "Expedia more expensive"}
                    print(f"    {meaning.get(idx, f'Value {idx}')}: {100*row['mean']:.2f}% booking (n={int(row['count'])})")
            
            # Aggregated competitor summary
            print("\n  AGGREGATED COMPETITOR SUMMARY:")
            rows_with_comp = self.df_train[self.df_train['comp1_rate'].notna()]
            if len(rows_with_comp) > 0:
                cheaper = rows_with_comp[rows_with_comp['comp1_rate'] == 1]['booking_bool'].mean()
                pricier = rows_with_comp[rows_with_comp['comp1_rate'] == -1]['booking_bool'].mean()
                same = rows_with_comp[rows_with_comp['comp1_rate'] == 0]['booking_bool'].mean()
                
                print(f"    Rows with comp data: {len(rows_with_comp)} ({100*len(rows_with_comp)/len(self.df_train):.1f}%)")
                print(f"    When Expedia cheaper: {100*cheaper:.2f}% booking")
                print(f"    When Expedia pricier: {100*pricier:.2f}% booking")
                print(f"    When same price: {100*same:.2f}% booking")
                uplift = ((cheaper - pricier) / pricier * 100) if pricier > 0 else 0
                print(f"    Uplift (cheaper vs pricier): {uplift:.1f}%")
                print(f"    → Worth engineering aggregated competitor feature if signal is strong")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            
            comp_for_plot = comp_stats.copy()
            comp_for_plot.index = comp_for_plot.index.astype(str)
            comp_for_plot['mean'].plot(kind='bar', ax=ax, color=['green', 'gray', 'red'][:len(comp_for_plot)])
            ax.set_xticklabels(['Cheaper (-1)', 'Same (0)', 'More Expensive (1)'][:len(comp_for_plot)], rotation=0)
            ax.set_ylabel('Booking Rate')
            ax.set_xlabel('Comp1 Rate Category')
            ax.set_title('Booking Rate by Competitor Price Signal')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('eda_10_competitor_signal.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_10_competitor_signal.png")
            plt.close()
    
    def visitor_history_segment(self):
        """Repeat customers vs non-repeat: do they book at different rates?"""
        print("\nVISITOR HISTORY SEGMENT ANALYSIS:")
        
        if 'visitor_hist_starrating' not in self.df_train.columns or 'booking_bool' not in self.df_train.columns:
            print("  Skipping: missing required columns")
            return
        
        repeat_users = self.df_train['visitor_hist_starrating'].notna()
        
        repeat_pct = 100 * repeat_users.mean()
        print(f"  Repeat customers (with history): {repeat_pct:.1f}%")
        print(f"  New customers (no history): {100 - repeat_pct:.1f}%")
        
        repeat_booking_rate = self.df_train[repeat_users]['booking_bool'].mean()
        new_booking_rate = self.df_train[~repeat_users]['booking_bool'].mean()
        
        print(f"  Repeat customer booking rate: {100*repeat_booking_rate:.2f}%")
        print(f"  New customer booking rate: {100*new_booking_rate:.2f}%")
        
        if new_booking_rate > 0:
            uplift = 100 * (repeat_booking_rate - new_booking_rate) / new_booking_rate
            print(f"  Uplift from repeat: {uplift:.1f}%")
            if abs(uplift) > 5:
                print(f"  → Justifies keeping visitor_hist columns despite {100 - repeat_pct:.1f}% missingness")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Segment sizes
        segments = ['Repeat\nCustomers', 'New\nCustomers']
        sizes = [repeat_pct, 100 - repeat_pct]
        axes[0].pie(sizes, labels=segments, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        axes[0].set_title('Customer Segmentation')
        
        # Booking rates by segment
        rates = [repeat_booking_rate * 100, new_booking_rate * 100]
        axes[1].bar(segments, rates, color=['skyblue', 'lightcoral'], edgecolor='black')
        axes[1].set_ylabel('Booking Rate (%)')
        axes[1].set_title('Booking Rate by Customer Type')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('eda_11_visitor_history_segment.png', dpi=100, bbox_inches='tight')
        print("  Saved: eda_11_visitor_history_segment.png")
        plt.close()
    
    def affinity_score_analysis(self):
        """Distribution of srch_query_affinity_score (93% missing, but predictive when present)."""
        print("\nSEARCH QUERY AFFINITY SCORE ANALYSIS:")
        
        if 'srch_query_affinity_score' not in self.df_train.columns:
            print("  Skipping: column not found")
            return
        
        affinity_present = self.df_train['srch_query_affinity_score'].notna()
        missing_pct = 100 * (1 - affinity_present.mean())
        
        print(f"  Missing: {missing_pct:.1f}%")
        print(f"  Present: {100 - missing_pct:.1f}%")
        
        if affinity_present.sum() > 0:
            affinity_values = self.df_train.loc[affinity_present, 'srch_query_affinity_score']
            print(f"  Min: {affinity_values.min():.4f}, Max: {affinity_values.max():.4f}")
            print(f"  Mean: {affinity_values.mean():.4f}, Median: {affinity_values.median():.4f}")
            
            # Correlation with booking (for non-null subset)
            if 'booking_bool' in self.df_train.columns:
                corr_with_booking = self.df_train.loc[affinity_present, 'srch_query_affinity_score'].corr(
                    self.df_train.loc[affinity_present, 'booking_bool']
                )
                print(f"  Correlation with booking (non-null): {corr_with_booking:.4f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(affinity_values, bins=30, color='mediumpurple', edgecolor='black')
            ax.set_xlabel('Affinity Score (log-probability)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Search Query Affinity Score Distribution (n={affinity_present.sum()}, {100 - missing_pct:.1f}% of data)')
            plt.tight_layout()
            plt.savefig('eda_12_affinity_score.png', dpi=100, bbox_inches='tight')
            print("  Saved: eda_12_affinity_score.png")
            plt.close()
    
    def position_bias_interpretation(self):
        """Explain position bias and metric choice (NDCG vs. classification)."""
        print("\nPOSITION BIAS & METRIC JUSTIFICATION:")
        
        if 'position' not in self.df_train.columns or 'random_bool' not in self.df_train.columns:
            print("  Skipping: missing position or random_bool")
            return
        
        if 'click_bool' not in self.df_train.columns:
            print("  Skipping: missing click_bool")
            return
        
        # Analyze random condition
        random_results = self.df_train[self.df_train['random_bool'] == 1]
        
        if len(random_results) > 0:
            random_stats = random_results.groupby('position')['click_bool'].agg(['mean', 'count']).reset_index()
            random_stats = random_stats[random_stats['position'] <= 15]
            
            if len(random_stats) > 0:
                click_at_1 = random_stats[random_stats['position'] == 1]['mean'].values
                click_at_10 = random_stats[random_stats['position'] == 10]['mean'].values
                
                if len(click_at_1) > 0 and len(click_at_10) > 0:
                    decay = (click_at_1[0] - click_at_10[0]) * 100
                    print(f"  Random condition (natural experiment):")
                    print(f"    Click rate at position 1: {click_at_1[0]*100:.1f}%")
                    print(f"    Click rate at position 10: {click_at_10[0]*100:.1f}%")
                    print(f"    Decay: {decay:.1f}pp (NOT due to algorithm)")
                    print(f"")
                    print(f"  KEY FINDING:")
                    print(f"    Position bias is INTRINSIC to user behavior (scroll fatigue).")
                    print(f"    Even randomly-ranked hotels show strong position decay.")
                    print(f"    This means:")
                    print(f"    1. Position is a STRONG feature (but confounded with quality)")
                    print(f"    2. A classification model (predict booking prob.) is insufficient")
                    print(f"    3. NDCG@5 is the right metric: rank high-quality items early")
                    print(f"    4. LambdaMART (listwise ranker) is ideal for this problem")
    
    # ============ Main Runner ============
    
    def run_full_eda(self, output_dir="plots/eda"):
        """Run complete EDA pipeline."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        os.chdir(output_path)
        
        self.load_data()
        
        # Phase 1
        self.basic_counts()
        self.position_availability()
        self.range_and_type_checks()
        self.competitor_missingness()
        self.missingness_overview()
        self.srch_id_overlap_analysis()
        
        # Phase 2
        self.target_variable_analysis()
        self.position_bias_analysis()
        self.property_level_features()
        self.price_analysis()
        self.search_level_features()
        
        # Phase 3
        self.correlation_analysis()
        self.propid_aggregates()
        self.within_query_features()
        
        # LambdaMART-specific analysis
        print("\n" + "="*60)
        print("LAMBDAMART-SPECIFIC ANALYSIS & FEATURE ENGINEERING PREP")
        print("="*60)
        self.propid_appearance_distribution()
        self.within_query_relative_plots()
        self.competitor_signal_analysis()
        self.visitor_history_segment()
        self.affinity_score_analysis()
        self.position_bias_interpretation()
        
        # Feature engineering checklist
        self._print_feature_engineering_checklist()
        
        print("\n" + "="*60)
        print("EDA COMPLETE — Ready for Feature Engineering")
        print("="*60)
    
    def _print_feature_engineering_checklist(self):
        """Print checklist of features to engineer for LambdaMART."""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING CHECKLIST (Next Phase)")
        print("="*60)
        print("""
PROP_ID AGGREGATES (✓ Already computed in prop_id_aggregates.csv):
  ✓ mean/std/median of all numeric features per prop_id
  ✓ mean booking_rate per prop_id
  → TODO: Apply Bayesian shrinkage to prop_id aggregates

WITHIN-QUERY RELATIVE FEATURES (✓ Analyzed, TODO: Engineer):
  ✓ price_rank_in_query (rank of price_usd within srch_id)
  ✓ starrating_rank_in_query (verified in plots)
  → TODO: Create derived features:
    - price_vs_prop_historical (price_usd - mean_price_per_prop_id)
    - price_pct_diff_from_prop_mean
    - location_score2_rank_in_query

MISSING INDICATORS (TODO: Engineer):
  → visitor_hist_missing (bool: visitor_hist_starrating is null)
  → orig_distance_missing (bool: orig_destination_distance is null)
  → affinity_score_missing (bool: srch_query_affinity_score is null)

COMPETITOR AGGREGATE (TODO: Engineer):
  → n_comps_where_expedia_cheaper (sum of comp_rate == 1 across comp1-8)
  → n_comps_where_expedia_pricier (sum of comp_rate == -1)
  → Handles missingness elegantly (treats null as 0)

EXPECTED OUTCOME:
  LambdaMART with ~60-80 features (numeric + within-query ranks)
  Training: pairwise ranking loss (LambdaRank objective)
  Evaluation: NDCG@5 (positions 1-5 are most important)
""")


def main(train_path, test_path=None, output_dir=None):
    """
    Run EDA from main.py
    
    Args:
        train_path: path to training CSV
        test_path: path to test CSV (optional)
        output_dir: directory to save plots (optional)
    """
    eda = HotelEDA(train_path, test_path)
    eda.run_full_eda(output_dir)


if __name__ == "__main__":
    # Example usage - find project root
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "Data" / "Raw" / "training_set_VU_DM.csv"
    test_path = project_root / "Data" / "Raw" / "test_set_VU_DM.csv"
    
    eda = HotelEDA(str(train_path), str(test_path))
    eda.run_full_eda(output_dir=str(project_root / "plots" / "eda"))
