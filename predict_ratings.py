"""
Yelp Restaurant Rating Predictor
A comprehensive machine learning project for predicting restaurant ratings using ensemble methods.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory setup
OUTPUT_DIR = '/mnt/user-data/outputs'
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = './outputs'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# Categorical feature columns
CATEGORICAL_FEATURES = [
    'GoodForKids', 'Alcohol', 'BusinessAcceptsCreditCards', 
    'WiFi', 'BikeParking', 'ByAppointmentOnly', 
    'WheelechairAccessible', 'OutdoorSeating', 
    'RestaurantsReservations', 'DogsAllowed', 'Caters'
]

# All features
ALL_FEATURES = ['review_count'] + CATEGORICAL_FEATURES

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return accuracy, tpr, fpr

def save_figure(filename):
    """Save matplotlib figure to output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(train_path='yelp242a_train.csv', 
                         test_path='yelp242a_test.csv'):
    """Load and prepare training and test datasets."""
    print_section("DATA LOADING AND PREPARATION")
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    
    # Create binary target variable
    print("\nCreating binary target variable (fourOrAbove)...")
    train_df['fourOrAbove'] = (train_df['stars'] >= 4).astype(int)
    test_df['fourOrAbove'] = (test_df['stars'] >= 4).astype(int)
    
    print(f"Class distribution (training):")
    print(f"  Stars < 4: {(train_df['fourOrAbove'] == 0).sum()} ({(train_df['fourOrAbove'] == 0).mean():.1%})")
    print(f"  Stars ≥ 4: {(train_df['fourOrAbove'] == 1).sum()} ({(train_df['fourOrAbove'] == 1).mean():.1%})")
    
    # Analyze missing values
    print("\nMissing value analysis:")
    for col in CATEGORICAL_FEATURES:
        missing_pct = (train_df[col] == '(Missing)').sum() / len(train_df) * 100
        if missing_pct > 0:
            print(f"  {col:30s}: {missing_pct:5.1f}%")
    
    # Prepare categorical variables
    print("\nEncoding categorical variables...")
    for col in CATEGORICAL_FEATURES:
        unique_vals = train_df[col].unique()
        if '(Missing)' in unique_vals:
            other_vals = [v for v in unique_vals if v != '(Missing)']
            ordered_cats = ['(Missing)'] + sorted(other_vals)
        else:
            ordered_cats = sorted(unique_vals)
        
        train_df[col] = pd.Categorical(train_df[col], categories=ordered_cats)
        test_df[col] = pd.Categorical(test_df[col], categories=ordered_cats)
    
    # Create feature matrices
    X_train = pd.get_dummies(train_df[ALL_FEATURES], drop_first=True)
    X_test = pd.get_dummies(test_df[ALL_FEATURES], drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    y_train = train_df['fourOrAbove']
    y_test = test_df['fourOrAbove']
    y_train_stars = train_df['stars']
    y_test_stars = test_df['stars']
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Total features after encoding: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, y_train_stars, y_test_stars, train_df, test_df

# ============================================================================
# BASELINE MODELS
# ============================================================================

def build_baseline_models(X_train, X_test, y_train, y_test, y_train_stars, y_test_stars):
    """Build and evaluate baseline classification models."""
    print_section("BASELINE MODELS")
    
    results = {}
    
    # 1. Linear Regression with Thresholding
    print_subsection("Linear Regression with Thresholding")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_stars)
    
    y_pred_lr_train = (lr_model.predict(X_train) >= 4).astype(int)
    y_pred_lr_test = (lr_model.predict(X_test) >= 4).astype(int)
    
    print(f"R² score: {lr_model.score(X_train, y_train_stars):.4f}")
    print(f"Training accuracy: {accuracy_score(y_train, y_pred_lr_train):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_lr_test):.4f}")
    
    results['Linear Regression'] = (lr_model, y_pred_lr_test)
    
    # 2. Logistic Regression
    print_subsection("Logistic Regression")
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    
    y_pred_logistic_test = logistic_model.predict(X_test)
    
    print(f"Training accuracy: {accuracy_score(y_train, logistic_model.predict(X_train)):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_logistic_test):.4f}")
    
    results['Logistic Regression'] = (logistic_model, y_pred_logistic_test)
    
    # 3. Decision Tree with Cross-Validation
    print_subsection("Decision Tree (CART) with Cross-Validation")
    
    tree_full = DecisionTreeClassifier(random_state=42)
    path = tree_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]
    
    print(f"Evaluating {len(ccp_alphas)} complexity parameters...")
    cv_scores = []
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    best_alpha = ccp_alphas[np.argmax(cv_scores)]
    print(f"Best ccp_alpha: {best_alpha:.6f}")
    print(f"Best CV accuracy: {max(cv_scores):.4f}")
    
    tree_model = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    tree_model.fit(X_train, y_train)
    
    y_pred_tree_test = tree_model.predict(X_test)
    
    print(f"Tree depth: {tree_model.get_depth()}")
    print(f"Number of leaves: {tree_model.get_n_leaves()}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_tree_test):.4f}")
    
    results['Decision Tree'] = (tree_model, y_pred_tree_test)
    
    # Visualize CV results
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, cv_scores, marker='o', markersize=3)
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best α = {best_alpha:.6f}')
    plt.xlabel('Complexity Parameter (ccp_alpha)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Decision Tree: Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure('cart_cv_results.png')
    plt.close()
    
    # Visualize decision tree
    plt.figure(figsize=(20, 12))
    plot_tree(tree_model, 
              feature_names=X_train.columns,
              class_names=['< 4 stars', '≥ 4 stars'],
              filled=True,
              fontsize=10,
              max_depth=3)
    plt.title('Decision Tree Structure (Top 3 Levels)', fontsize=16)
    save_figure('decision_tree.png')
    plt.close()
    
    return results

# ============================================================================
# ENSEMBLE MODELS
# ============================================================================

def build_ensemble_models(X_train, X_test, y_train, y_test, baseline_results):
    """Build and evaluate ensemble models."""
    print_section("ENSEMBLE MODELS")
    
    results = {}
    
    # 1. Voting Ensemble
    print_subsection("Majority Voting Ensemble")
    votes = np.column_stack([pred for _, pred in baseline_results.values()])
    y_pred_voting = stats.mode(votes, axis=1, keepdims=False).mode
    
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_voting):.4f}")
    results['Voting Ensemble'] = (None, y_pred_voting)
    
    # 2. Bagging (Vanilla)
    print_subsection("Bagging (Vanilla)")
    bagging_model = RandomForestClassifier(
        n_estimators=100,
        max_features=X_train.shape[1],
        random_state=42,
        n_jobs=-1
    )
    bagging_model.fit(X_train, y_train)
    y_pred_bagging = bagging_model.predict(X_test)
    
    print(f"Training accuracy: {accuracy_score(y_train, bagging_model.predict(X_train)):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_bagging):.4f}")
    results['Bagging'] = (bagging_model, y_pred_bagging)
    
    # 3. Random Forest with Grid Search
    print_subsection("Random Forest with Hyperparameter Tuning")
    
    param_grid = {
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'n_estimators': [50, 100, 200],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search_rf = GridSearchCV(rf_model, param_grid, cv=5, 
                                   scoring='accuracy', n_jobs=-1, verbose=1)
    
    print("Running grid search (this may take a few minutes)...")
    grid_search_rf.fit(X_train, y_train)
    
    rf_best = grid_search_rf.best_estimator_
    y_pred_rf = rf_best.predict(X_test)
    
    print(f"\nBest parameters: {grid_search_rf.best_params_}")
    print(f"Best CV accuracy: {grid_search_rf.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    results['Random Forest'] = (rf_best, y_pred_rf)
    
    # Visualize RF CV results
    cv_results = pd.DataFrame(grid_search_rf.cv_results_)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    max_feat_results = cv_results.groupby('param_max_features')['mean_test_score'].agg(['mean', 'std'])
    max_feat_results.plot(kind='bar', y='mean', yerr='std', ax=axes[0], capsize=4)
    axes[0].set_xlabel('max_features')
    axes[0].set_ylabel('CV Accuracy')
    axes[0].set_title('Random Forest: max_features Impact')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Mean CV Accuracy'])
    
    n_est_results = cv_results.groupby('param_n_estimators')['mean_test_score'].agg(['mean', 'std'])
    n_est_results.plot(kind='bar', y='mean', yerr='std', ax=axes[1], capsize=4)
    axes[1].set_xlabel('n_estimators')
    axes[1].set_ylabel('CV Accuracy')
    axes[1].set_title('Random Forest: n_estimators Impact')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Mean CV Accuracy'])
    
    plt.tight_layout()
    save_figure('rf_cv_results.png')
    plt.close()
    
    # 4. Gradient Boosting
    print_subsection("Gradient Boosting with Hyperparameter Tuning")
    
    param_grid_gb = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_leaf_nodes': [3, 5, 10, 15, 20, 30]
    }
    
    gb_model = GradientBoostingClassifier(learning_rate=0.1, random_state=42)
    grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=5, 
                                   scoring='accuracy', n_jobs=-1, verbose=1)
    
    print("Running grid search (this may take several minutes)...")
    grid_search_gb.fit(X_train, y_train)
    
    gb_best = grid_search_gb.best_estimator_
    y_pred_gb = gb_best.predict(X_test)
    
    print(f"\nBest parameters: {grid_search_gb.best_params_}")
    print(f"Best CV accuracy: {grid_search_gb.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    
    results['Gradient Boosting'] = (gb_best, y_pred_gb)
    
    # Visualize GB CV results as heatmap
    cv_results_gb = pd.DataFrame(grid_search_gb.cv_results_)
    pivot = cv_results_gb.pivot_table(values='mean_test_score',
                                      index='param_max_leaf_nodes',
                                      columns='param_n_estimators')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'CV Accuracy'})
    plt.xlabel('n_estimators (Number of Trees)')
    plt.ylabel('max_leaf_nodes (Tree Complexity)')
    plt.title('Gradient Boosting: Bias-Variance Tradeoff\nCV Accuracy Heatmap')
    save_figure('gb_cv_heatmap.png')
    plt.close()
    
    return results, rf_best, gb_best

# ============================================================================
# MODEL EVALUATION AND COMPARISON
# ============================================================================

def evaluate_models(all_results, y_test):
    """Evaluate and compare all models."""
    print_section("MODEL PERFORMANCE COMPARISON")
    
    comparison_data = []
    for model_name, (_, y_pred) in all_results.items():
        acc, tpr, fpr = calculate_metrics(y_test, y_pred)
        comparison_data.append({
            'Model': model_name,
            'Accuracy': acc,
            'TPR (Sensitivity)': tpr,
            'FPR': fpr
        })
    
    results_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    print("\nTest Set Performance:\n")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_comparison.csv'), index=False)
    print(f"\nSaved results to: model_performance_comparison.csv")
    
    # Print best model
    best = results_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best['Model']}")
    print(f"   Accuracy: {best['Accuracy']:.4f}")
    print(f"   TPR: {best['TPR (Sensitivity)']:.4f}")
    print(f"   FPR: {best['FPR']:.4f}")
    
    return results_df

# ============================================================================
# FEATURE IMPORTANCE AND INSIGHTS
# ============================================================================

def analyze_feature_importance(rf_model, gb_model, X_train):
    """Analyze and visualize feature importance."""
    print_section("FEATURE IMPORTANCE ANALYSIS")
    
    rf_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    gb_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features (Random Forest):")
    print(rf_importance.head(10).to_string(index=False))
    
    print("\nTop 10 Features (Gradient Boosting):")
    print(gb_importance.head(10).to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    top_n = 15
    for ax, importance, title in zip(axes, 
                                      [rf_importance, gb_importance],
                                      ['Random Forest', 'Gradient Boosting']):
        ax.barh(range(top_n), importance['Importance'].head(top_n))
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(importance['Feature'].head(top_n))
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{title}: Top {top_n} Features')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure('feature_importance.png')
    plt.close()

def generate_business_insights(train_df):
    """Generate actionable business insights."""
    print_section("ACTIONABLE BUSINESS INSIGHTS")
    
    print("\n📊 Three Key Recommendations for Restaurant Owners:\n")
    
    # Insight 1: Review Count
    avg_reviews = train_df.groupby('fourOrAbove')['review_count'].mean()
    print("1️⃣  ENCOURAGE CUSTOMER REVIEWS")
    print(f"   → Restaurants with ≥4 stars have {avg_reviews[1]:.0f} reviews on average")
    print(f"   → Restaurants with <4 stars have {avg_reviews[0]:.0f} reviews on average")
    print(f"   → Difference: {avg_reviews[1] - avg_reviews[0]:.0f} more reviews")
    print("   💡 ACTION: Actively ask satisfied customers to leave reviews")
    
    # Insight 2: WiFi
    wifi_rating = train_df.groupby('WiFi')['fourOrAbove'].mean().sort_values(ascending=False)
    print("\n2️⃣  OFFER FREE WIFI")
    print("   → Proportion of high-rated restaurants by WiFi:")
    for wifi_type, prop in wifi_rating.items():
        print(f"      {wifi_type:15s}: {prop:.1%}")
    print("   💡 ACTION: Install and advertise free WiFi for customers")
    
    # Insight 3: Full Bar + Reservations
    train_df['full_bar'] = (train_df['Alcohol'] == "'full_bar'").astype(int)
    train_df['takes_reservations'] = (train_df['RestaurantsReservations'] == 'TRUE').astype(int)
    
    combo = train_df.groupby(train_df['full_bar'] & train_df['takes_reservations'])['fourOrAbove'].mean()
    print("\n3️⃣  ENHANCE SERVICE OPTIONS")
    print(f"   → Restaurants with BOTH full bar AND reservations: {combo.iloc[-1]:.1%} high-rated")
    print(f"   → Restaurants without both: {combo.iloc[0]:.1%} high-rated")
    print("   💡 ACTION: Consider offering full bar service and accepting reservations")
    
    # Visualize insights
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Review count distribution
    axes[0, 0].hist([train_df[train_df['fourOrAbove']==0]['review_count'],
                     train_df[train_df['fourOrAbove']==1]['review_count']],
                    bins=30, label=['< 4 stars', '≥ 4 stars'], alpha=0.7)
    axes[0, 0].set_xlabel('Review Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Review Count Distribution by Rating')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 500)
    
    # WiFi
    wifi_data = train_df.groupby('WiFi')['fourOrAbove'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(wifi_data)), wifi_data.values, color='skyblue')
    axes[0, 1].set_xticks(range(len(wifi_data)))
    axes[0, 1].set_xticklabels(wifi_data.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Proportion with ≥4 Stars')
    axes[0, 1].set_title('WiFi Type vs High Ratings')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alcohol
    alcohol_data = train_df.groupby('Alcohol')['fourOrAbove'].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(alcohol_data)), alcohol_data.values, color='coral')
    axes[1, 0].set_xticks(range(len(alcohol_data)))
    axes[1, 0].set_xticklabels(alcohol_data.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Proportion with ≥4 Stars')
    axes[1, 0].set_title('Alcohol Service vs High Ratings')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combo effect
    axes[1, 1].bar(['Without Both', 'With Both'], 
                   [combo.iloc[0], combo.iloc[-1]], 
                   color=['coral', 'seagreen'])
    axes[1, 1].set_ylabel('Proportion with ≥4 Stars')
    axes[1, 1].set_title('Combined Effect: Full Bar + Reservations')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure('actionable_insights.png')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("🍽️  YELP RESTAURANT RATING PREDICTOR")
    print("="*80)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, y_train_stars, y_test_stars, train_df, test_df = \
        load_and_prepare_data()
    
    # Build baseline models
    baseline_results = build_baseline_models(
        X_train, X_test, y_train, y_test, y_train_stars, y_test_stars
    )
    
    # Build ensemble models
    ensemble_results, rf_best, gb_best = build_ensemble_models(
        X_train, X_test, y_train, y_test, baseline_results
    )
    
    # Combine all results
    all_results = {**baseline_results, **ensemble_results}
    
    # Evaluate models
    evaluate_models(all_results, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(rf_best, gb_best, X_train)
    
    # Generate business insights
    generate_business_insights(train_df)
    
    # Final summary
    print_section("✅ ANALYSIS COMPLETE")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  📊 cart_cv_results.png")
    print("  🌳 decision_tree.png")
    print("  📈 rf_cv_results.png")
    print("  🔥 gb_cv_heatmap.png")
    print("  ⭐ feature_importance.png")
    print("  💡 actionable_insights.png")
    print("  📋 model_performance_comparison.csv")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
