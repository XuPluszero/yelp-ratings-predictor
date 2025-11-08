# Example Output

This document shows what to expect when running the analysis.

## Console Output

When you run `python predict_ratings.py`, you'll see:

```
================================================================================
🍽️  YELP RESTAURANT RATING PREDICTOR
================================================================================

================================================================================
DATA LOADING AND PREPARATION
================================================================================

Loading datasets...
Training set: 6272 samples
Test set: 2688 samples

Creating binary target variable (fourOrAbove)...
Class distribution (training):
  Stars < 4: 3491 (55.7%)
  Stars ≥ 4: 2781 (44.3%)

Missing value analysis:
  GoodForKids                   :  29.9%
  Alcohol                       :  34.2%
  WiFi                          :  32.2%
  BikeParking                   :  29.4%
  ...

Feature matrix shape: (6272, 25)
Total features after encoding: 25

================================================================================
BASELINE MODELS
================================================================================

--- Linear Regression with Thresholding ---
R² score: 0.1627
Training accuracy: 0.6183
Test accuracy: 0.6347

--- Logistic Regression ---
Training accuracy: 0.6688
Test accuracy: 0.6778

--- Decision Tree (CART) with Cross-Validation ---
Evaluating 915 complexity parameters...
Best ccp_alpha: 0.000579
Best CV accuracy: 0.6663
Tree depth: 11
Number of leaves: 42
Test accuracy: 0.6656

================================================================================
ENSEMBLE MODELS
================================================================================

--- Majority Voting Ensemble ---
Test accuracy: 0.6733

--- Bagging (Vanilla) ---
Training accuracy: 0.9956
Test accuracy: 0.6741

--- Random Forest with Hyperparameter Tuning ---
Running grid search (this may take a few minutes)...

Best parameters: {'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 200}
Best CV accuracy: 0.6789
Test accuracy: 0.6850

--- Gradient Boosting with Hyperparameter Tuning ---
Running grid search (this may take several minutes)...

Best parameters: {'max_leaf_nodes': 15, 'n_estimators': 200}
Best CV accuracy: 0.6802
Test accuracy: 0.6854

================================================================================
MODEL PERFORMANCE COMPARISON
================================================================================

Test Set Performance:

                   Model  Accuracy  TPR (Sensitivity)       FPR
      Gradient Boosting    0.6854            0.6977    0.3254
          Random Forest    0.6850            0.6956    0.3254
    Logistic Regression    0.6778            0.6694    0.3129
        Voting Ensemble    0.6733            0.6816    0.3348
                Bagging    0.6741            0.6838    0.3365
        Decision Tree     0.6656            0.6618    0.3311
  Linear Regression       0.6347            0.6367    0.3672

🏆 BEST MODEL: Gradient Boosting
   Accuracy: 0.6854
   TPR: 0.6977
   FPR: 0.3254

================================================================================
FEATURE IMPORTANCE ANALYSIS
================================================================================

Top 10 Features (Random Forest):
         Feature  Importance
   review_count      0.5234
    Alcohol_none      0.0823
 WiFi_'free'          0.0612
...

================================================================================
ACTIONABLE BUSINESS INSIGHTS
================================================================================

📊 Three Key Recommendations for Restaurant Owners:

1️⃣  ENCOURAGE CUSTOMER REVIEWS
   → Restaurants with ≥4 stars have 142 reviews on average
   → Restaurants with <4 stars have 78 reviews on average
   → Difference: 64 more reviews
   💡 ACTION: Actively ask satisfied customers to leave reviews

2️⃣  OFFER FREE WIFI
   → Proportion of high-rated restaurants by WiFi:
      'free'          : 48.2%
      'no'            : 42.1%
      'paid'          : 38.5%
   💡 ACTION: Install and advertise free WiFi for customers

3️⃣  ENHANCE SERVICE OPTIONS
   → Restaurants with BOTH full bar AND reservations: 52.3% high-rated
   → Restaurants without both: 41.8% high-rated
   💡 ACTION: Consider offering full bar service and accepting reservations

================================================================================
✅ ANALYSIS COMPLETE
================================================================================

All results saved to: ./outputs/

Generated files:
  📊 cart_cv_results.png
  🌳 decision_tree.png
  📈 rf_cv_results.png
  🔥 gb_cv_heatmap.png
  ⭐ feature_importance.png
  💡 actionable_insights.png
  📋 model_performance_comparison.csv

================================================================================
```

## Generated Visualizations

### 1. `cart_cv_results.png`
Shows the cross-validation accuracy across different complexity parameters for the decision tree, helping identify the optimal pruning threshold.

### 2. `decision_tree.png`
Visual representation of the pruned decision tree (top 3 levels), showing the most important decision rules.

### 3. `rf_cv_results.png`
Displays how different hyperparameters (max_features, n_estimators) affect Random Forest performance during cross-validation.

### 4. `gb_cv_heatmap.png`
Heatmap showing the bias-variance tradeoff in Gradient Boosting, with tree complexity on one axis and number of trees on the other.

### 5. `feature_importance.png`
Side-by-side comparison of feature importance from Random Forest and Gradient Boosting models.

### 6. `actionable_insights.png`
Four-panel visualization showing:
- Review count distribution by rating
- WiFi type vs high ratings
- Alcohol service vs high ratings
- Combined effect of full bar + reservations

### 7. `model_performance_comparison.csv`
CSV table with detailed performance metrics for all models, including accuracy, TPR, and FPR.

## Typical Runtime

- **Data loading and preparation**: < 1 second
- **Baseline models**: 5-10 seconds
- **Random Forest grid search**: 2-5 minutes
- **Gradient Boosting grid search**: 5-10 minutes
- **Total runtime**: ~10-15 minutes

Runtime varies based on your hardware (CPU cores, RAM).
