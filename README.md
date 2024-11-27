# Employee Satisfaction and Churn Prediction Dashboard

## Aim

The aim of this project is to predict employee churn and evaluate satisfaction levels using machine learning techniques. By leveraging key organizational data, the project seeks to provide actionable insights to help HR departments mitigate turnover risks.

### Streamlit App Overview

![Streamlit App](https://github.com/Adityalolla04/End-to-End-Project-on-HR-Churn-Data/blob/main/Images/Sreamlit%20App.png "Streamlit Application Interface")


## Description

This project utilizes a comprehensive dataset to analyze employee behavior and identify factors that contribute to satisfaction and churn. An interactive dashboard was developed using Streamlit, allowing users to input data, view predictions, and explore visual insights.

Key components:
- **Churn Prediction**: Assessing whether an employee is likely to leave the organization.
- **Data Visualization**: Highlighting trends and patterns in employee behavior.
- **Insights**: Offering data-driven guidance to improve retention strategies.

The analysis enables organizations to proactively address factors influencing employee turnover.

## Data Preprocessing

The dataset `hr_employee_churn_data.csv` includes key features for predicting churn. Preprocessing steps:
1. **Handling Missing Values**: Ensured completeness of the dataset.
2. **Encoding Categorical Variables**: Converted `salary` into numerical values (`low`, `medium`, `high`).
3. **Feature Scaling**: Normalized continuous variables for compatibility with the model.

This ensures the data is ready for building a robust predictive model.

## Model Training

The XGBoost classifier was trained to predict churn using:
- **Training and Testing Data**: Split the dataset into training (80%) and testing (20%) subsets.
- **Hyperparameter Tuning**: Optimized parameters such as learning rate and max depth.
- **Evaluation Metrics**: Accuracy, precision, recall, and feature importance were used to assess performance.

### Model Performance Metrics
- ** Accuracy Score**: *[98.23]*
- **Feature Importance**: Identified satisfaction level, time spent at the company, number of projects, and average monthly hours as key predictors.

## Churn Prediction

The model predicts churn based on input features:
- Satisfaction Level
- Last Evaluation
- Number of Projects
- Average Monthly Hours
- Time Spent at Company
- Work Accident
- Promotion in the Last 5 Years
- Salary Level (Low, Medium, High)

**Prediction Outputs**:
- **Churn**: Employee is likely to leave the organization.
- **No Churn**: Employee is likely to stay.

## Data Visualization

Key visual insights include:
1. **Satisfaction Levels**:
   - Churned employees often exhibit low satisfaction.
   - Higher satisfaction correlates with employee retention.
### Satisfaction Level

![Satisfaction Level Distribution](https://github.com/Adityalolla04/End-to-End-Project-on-HR-Churn-Data/blob/main/Images/Satisfaction%20level.png "Satisfaction Level Distribution")

2. **Last Evaluation Scores**:
   - Extreme scores (high or low) are common among churned employees.
   - Steady performance reflects in non-churned employees.

### Last Evaluation Scores

![Last Evaluation Distribution](https://github.com/Adityalolla04/End-to-End-Project-on-HR-Churn-Data/blob/main/Images/last%20evaluation.png "Last Evaluation Distribution")

3. **Number of Projects**:
   - Imbalance in project workload (too few or too many) correlates with churn.
   - Moderate workloads support retention.

### Number of Projects

![Number of Projects Distribution](https://github.com/Adityalolla04/End-to-End-Project-on-HR-Churn-Data/blob/main/Images/Number%20of%20project.png "Number of Projects Distribution")

3. **Correlation Heatmap**:
   -There is a strong negative correlation between employee satisfaction level and churn (-0.13), indicating that employees with lower satisfaction levels are more likely to leave the organization. Improving workplace satisfaction can significantly reduce churn rates.
   -Workload factors such as excessive projects, high monthly hours, and prolonged tenure show moderate correlations with churn, suggesting the need for balanced workloads and clear career growth opportunities to retain employees

### Correlation Heatmap

![Correlation Heatmap](https://github.com/Adityalolla04/End-to-End-Project-on-HR-Churn-Data/blob/main/Images/Heatmap.png "Correlation Heatmap")


## Key Insights

1. **Employee Satisfaction**: The most critical factor for retention. Improving satisfaction levels can significantly reduce churn.
2. **Workload Management**: Balanced workloads prevent underutilization or burnout, reducing churn risks.
3. **Performance Engagement**: Employees with consistent evaluation scores tend to stay, highlighting the importance of regular feedback.
4. **Targeted Interventions**: These insights provide HR professionals with actionable data to design effective retention programs.

## Conclusion

This project demonstrates the value of machine learning in addressing employee churn. By identifying the factors that influence satisfaction and turnover, organizations can implement data-driven retention strategies.

**Key Takeaways**:
- Employee satisfaction is the strongest predictor of churn, emphasizing the need for a positive work environment.
- Balanced workloads are essential to reducing turnover risks.
- Predictive analytics enable proactive intervention for at-risk employees.
