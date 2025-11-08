# Yelp Restaurant Rating Predictor 🍽️⭐

A comprehensive machine learning project for predicting restaurant ratings on Yelp using ensemble methods and analyzing key factors that influence customer satisfaction.

## 📋 Overview

This project analyzes Yelp restaurant data from Las Vegas, Nevada to build predictive models that classify whether a restaurant will receive high ratings (4+ stars) based on various business attributes. The project implements and compares multiple machine learning approaches, from simple baselines to advanced ensemble methods.

## ✨ Key Features

- **Multiple ML Models**: Implements 7 different classification approaches
  - Linear Regression with thresholding
  - Logistic Regression
  - Decision Trees (CART) with cross-validation
  - Voting Ensemble
  - Bagging
  - Random Forest
  - Gradient Boosting

- **Comprehensive Analysis**:
  - Feature importance visualization
  - Bias-variance tradeoff analysis
  - Performance comparison across all models
  - Actionable business insights for restaurant owners

- **Professional Visualizations**:
  - Cross-validation results
  - Decision tree diagrams
  - Feature importance charts
  - Business insight visualizations

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yelp-ratings-predictor.git
cd yelp-ratings-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your data files in the project directory:
   - `yelp242a_train.csv`
   - `yelp242a_test.csv`

### Usage

Run the main analysis script:
```bash
python predict_ratings.py
```

The script will:
1. Load and preprocess the data
2. Build and evaluate all models
3. Generate visualizations and save them to the `outputs/` directory
4. Print detailed results and insights to the console

## 📊 Dataset

The dataset contains information about restaurants in Las Vegas with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| **stars** | Average star rating (1-5) | Continuous |
| **review_count** | Number of reviews | Integer |
| **GoodForKids** | Kid-friendly status | Categorical |
| **Alcohol** | Alcohol service type | Categorical |
| **WiFi** | WiFi availability | Categorical |
| **BikeParking** | Bike parking availability | Categorical |
| **OutdoorSeating** | Outdoor seating availability | Categorical |
| **RestaurantsReservations** | Accepts reservations | Categorical |
| **Caters** | Provides catering | Categorical |
| *...and more* | Additional business attributes | Categorical |

**Target Variable**: `fourOrAbove` - Binary indicator (1 if stars ≥ 4, 0 otherwise)

## 🎯 Model Performance

| Model | Test Accuracy | TPR | FPR |
|-------|--------------|-----|-----|
| **Gradient Boosting** | ~0.68 | ~0.70 | ~0.32 |
| **Random Forest** | ~0.68 | ~0.69 | ~0.32 |
| **Logistic Regression** | ~0.68 | ~0.67 | ~0.31 |
| **Voting Ensemble** | ~0.67 | ~0.68 | ~0.33 |
| **Decision Tree** | ~0.67 | ~0.66 | ~0.32 |
| **Bagging** | ~0.67 | ~0.68 | ~0.33 |
| **Linear Reg (Threshold)** | ~0.63 | ~0.64 | ~0.37 |

*Note: Exact results may vary based on random seed and data split*

## 🔑 Key Insights

### Top Predictive Features:
1. **review_count** - Most important predictor across all models
2. **WiFi availability** - Free WiFi correlates with higher ratings
3. **Alcohol service** - Full bar service associated with better ratings
4. **Reservations** - Accepting reservations signals quality
5. **Business attributes** - Various amenities contribute to ratings

### Actionable Recommendations for Restaurants:

1. **Encourage Customer Reviews**
   - Higher-rated restaurants have significantly more reviews
   - Active review solicitation can improve visibility and ratings

2. **Offer Free WiFi**
   - Restaurants with free WiFi show higher average ratings
   - Modern amenity that customers expect

3. **Enhance Service Options**
   - Consider offering full bar service
   - Implement a reservation system
   - Combined effect shows strong positive impact

## 📁 Project Structure

```
yelp-ratings-predictor/
│
├── predict_ratings.py          # Main analysis script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore file
│
├── yelp242a_train.csv         # Training data (not included)
├── yelp242a_test.csv          # Test data (not included)
│
└── outputs/                    # Generated results
    ├── cart_cv_results.png
    ├── decision_tree.png
    ├── rf_cv_results.png
    ├── gb_cv_heatmap.png
    ├── feature_importance.png
    ├── actionable_insights.png
    └── model_performance_comparison.csv
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Seaborn** - Statistical visualization
- **SciPy** - Scientific computing

## 📈 Methodology

1. **Data Preparation**
   - Handle missing values by treating them as explicit categories
   - Create binary target variable (fourOrAbove)
   - One-hot encode categorical features

2. **Model Development**
   - Implement baseline models (Linear/Logistic Regression, Decision Tree)
   - Build ensemble models (Voting, Bagging, Random Forest, Gradient Boosting)
   - Perform cross-validation for hyperparameter tuning

3. **Evaluation**
   - Compare models using accuracy, TPR, and FPR
   - Analyze feature importance
   - Generate business insights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from the [Yelp Dataset](https://www.yelp.com/dataset)
- Inspired by the importance of data-driven decision making in the hospitality industry
- Built with best practices from the machine learning community

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/yelp-ratings-predictor](https://github.com/yourusername/yelp-ratings-predictor)

---

⭐ Star this repo if you found it helpful!
