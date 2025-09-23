# Supermarket Sales Prediction 🛒📊

A comprehensive machine learning project that predicts supermarket sales using multiple regression models and compares their performance with modern Large Language Models.


## Project Overview

This project explores a dataset of **8,523 supermarket sales records** and implements multiple machine learning models to predict sales based on various features related to products, customers, and store operations.

### Problem Statement
Develop accurate sales prediction models to help supermarkets optimize inventory management, revenue planning, and business decision-making through data-driven insights.

## Dataset Information

- **Source**: Big Market Sales from Kaggle
- **Size**: 8,523 sales records
- **Features**: Product details, customer information, store operations data

### Key Dataset Features
- Product ID and details
- Outlet information and establishment year
- Product visibility metrics  
- Maximum Retail Price (MRP)
- Sales figures (target variable)

## Machine Learning Models Implemented

### 1. Linear Regression (Baseline Model)
- **Purpose**: Establish performance baseline for comparison
- **Implementation**: Custom gradient descent with 1000 iterations
- **Features**: Manual bias term addition, float64 normalization
- **Learning Rate**: 0.001
- **Training Method**: Batch Gradient Descent

### 2. Random Forest Regressor
- **Purpose**: Capture non-linear relationships in data
- **Configuration**: 100 estimators (optimal performance)
- **Performance**: Underperformed due to dataset complexity limitations
- **Insight**: Model struggled with sharp variations in sales data

### 3. Gradient Boosting (Best Performer)
- **Purpose**: Incremental improvement through sequential error correction
- **Configuration**: 100 estimators, learning rate 0.1
- **Training Time**: ~20 minutes
- **Advantage**: Superior handling of outliers and non-linear patterns

## Data Preprocessing Pipeline

### 1. Data Cleaning
- Identified and filled missing values with appropriate substitutions
- Removed irrelevant features (ProductID, OutletID) that don't contribute to sales prediction

### 2. Feature Engineering
- **Outlet Age Calculation**: Replaced establishment year with `OutletAge = CurrentYear - EstablishmentYear`
- **Log Transformation**: Applied `np.log1p()` to ProductVisibility for normalization
- **Feature Combination**: Created `MRP_weight` feature to capture combined relationships
- **Categorical Data Processing**: Analyzed unique values and standardized categories

### 3. Data Preparation
- Split dataset into training and testing sets
- Normalized features for stable model training
- Ensured data types compatibility across all models

## Performance Evaluation

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Measures average prediction error magnitude (lower is better)
- **R² Score**: Indicates proportion of variance explained by the model (higher is better)

### Model Performance Comparison

| Model | RMSE | R² Score | Performance |
|-------|------|----------|-------------|
| Linear Regression | Higher | Lower | Baseline (underfitting) |
| Random Forest | Higher | ~0.33 | Struggled with complexity |
| **Gradient Boosting** | **1041.76** | **0.600** | **Best Performance** |
| LLM Comparison | 1079.47 | 0.571 | Outperformed by Gradient Boosting |

## LLM Comparison Analysis

### Methodology
- Compared best-performing Gradient Boosting model with Large Language Model
- Used representative subset of test data for structured prompts
- Generated sales predictions using same feature set
- Conducted direct performance comparison

### Results
- **Gradient Boosting** achieved **3.5% better RMSE** than LLM
- Demonstrated superiority of traditional ML for structured tabular data
- Validated model selection approach through modern AI comparison

## Key Insights & Conclusions

### Model Analysis
1. **Linear Regression**: Underfitted due to simplicity, couldn't capture complex relationships
2. **Random Forest**: Struggled with sharp sales variations, performed worse than expected
3. **Gradient Boosting**: Successfully handled non-linear patterns through sequential error correction

### Best Model Performance
- **R² = 0.60**: Explains 60% of sales variance
- **RMSE = 1041.7**: Maintains reasonable prediction accuracy
- **Superiority**: Outperformed both traditional models and modern LLM approaches

## Technical Implementation

### Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Scikit-learn**: Model comparison and validation
- **Custom Implementations**: Built models from scratch for deeper understanding
- **Statistical Analysis**: Comprehensive model evaluation and comparison

### Development Approach
- **Collaborative Learning**: Team-based exploration of regression algorithms
- **Iterative Development**: Multiple 2.5-hour sprint sessions
- **From-Scratch Implementation**: Built understanding through manual coding
- **Cross-Validation**: Rigorous testing and performance evaluation

## Team Collaboration

### Working Methodology
- **Distributed Learning**: Each member led different model implementations
- **Collaborative Integration**: Worked together on all project phases
- **Knowledge Sharing**: Regular discussions on model performance and interpretation
- **Comprehensive Coverage**: From data preprocessing to final evaluation

### Session Structure
- **Duration**: Multiple 2.5-hour intensive sessions
- **Focus Areas**: Model understanding, implementation, and performance analysis
- **Outcome**: Enhanced practical understanding of regression algorithms

## Business Impact

### Practical Applications
- **Inventory Management**: Accurate sales forecasting for stock optimization
- **Revenue Planning**: Data-driven business decision making
- **Resource Allocation**: Improved operational efficiency through predictive insights
- **Market Analysis**: Understanding sales patterns and customer behavior

## Future Enhancements

### Potential Improvements
- **Feature Engineering**: Additional domain-specific features
- **Model Ensemble**: Combining multiple models for better performance  
- **Hyperparameter Optimization**: Advanced tuning techniques
- **Real-time Prediction**: Implementation of streaming prediction pipeline
- **Deep Learning**: Exploring neural network approaches



---

*Developed as part of Machine Learning coursework, demonstrating practical application of regression algorithms and modern AI comparison techniques.*
