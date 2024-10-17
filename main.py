import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the pre-trained model and dataset
model = joblib.load('models/rf_churn_model.pkl')
data = pd.read_csv('hr_employee_churn_data.csv')

# Convert categorical columns to numeric for correlation and model compatibility
data['salary'] = data['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# Page layout
st.title("Employee Churn Prediction Dashboard")

# Collect user inputs for the features
st.header("Employee Input Features")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, value=0.2)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, value=0.9)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=7)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=0, max_value=300, value=270)
time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=6)
salary = st.selectbox("Salary Level", ["low", "medium", "high"], index=0)
work_accident = st.selectbox("Work Accident", [0, 1], index=0)
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], index=0)

# Create a DataFrame from the input data
user_input = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,  # Keep consistent with training typo
    'time_spend_company': time_spend_company,
    'Work_accident': work_accident,
    'promotion_last_5years': promotion_last_5years,
    'salary': salary
}

user_df = pd.DataFrame([user_input])

# Manually encode salary to match the model's training data
if salary == "medium":
    user_df['salary_1'] = 1
    user_df['salary_2'] = 0
elif salary == "high":
    user_df['salary_1'] = 0
    user_df['salary_2'] = 1
else:
    user_df['salary_1'] = 0
    user_df['salary_2'] = 0

# Drop the original salary column
user_df = user_df.drop('salary', axis=1)

# Ensure that the DataFrame has the same column order as the training data
required_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                    'average_montly_hours', 'time_spend_company', 'Work_accident', 
                    'promotion_last_5years', 'salary_1', 'salary_2']

for col in required_columns:
    if col not in user_df.columns:
        user_df[col] = 0

# Reorder the DataFrame columns to match the order of the training data
user_df = user_df[required_columns]

# Predict churn using the model
if st.button("Predict Churn"):
    prediction = model.predict(user_df)
    if prediction[0] == 1:
        st.write("### The predicted churn result is: **Churn**")
        
        # Display graphs and summary for Churned employees
        churn_data = data[data['left'] == 1]
        
        st.write("#### Churned Employees: Key Insights")
        
        # Satisfaction Level Distribution for Churned Employees
        st.write("**Satisfaction Level for Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.histplot(data=churn_data, x='satisfaction_level', kde=True)
        st.pyplot(plt.gcf())

        # Last Evaluation Distribution for Churned Employees
        st.write("**Last Evaluation for Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.histplot(data=churn_data, x='last_evaluation', kde=True)
        st.pyplot(plt.gcf())

        # Number of Projects for Churned Employees
        st.write("**Number of Projects for Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.countplot(x='number_project', data=churn_data)
        st.pyplot(plt.gcf())

        # Summary of Churn Data
        st.write("""
        **Summary for Churned Employees**:  
        - Churned employees often have a combination of low satisfaction levels and high last evaluations.  
        - Overworking, represented by higher monthly hours and higher project loads, is often associated with churn.
        - The XGBoost classifier effectively identifies churn patterns based on these factors and helps predict which employees are at risk of leaving.
        """)

    else:
        st.write("### The predicted churn result is: **No Churn**")
        
        # Display graphs and summary for Non-Churned employees
        no_churn_data = data[data['left'] == 0]
        
        st.write("#### Non-Churned Employees: Key Insights")
        
        # Satisfaction Level Distribution for Non-Churned Employees
        st.write("**Satisfaction Level for Non-Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.histplot(data=no_churn_data, x='satisfaction_level', kde=True)
        st.pyplot(plt.gcf())

        # Last Evaluation Distribution for Non-Churned Employees
        st.write("**Last Evaluation for Non-Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.histplot(data=no_churn_data, x='last_evaluation', kde=True)
        st.pyplot(plt.gcf())

        # Number of Projects for Non-Churned Employees
        st.write("**Number of Projects for Non-Churned Employees**")
        plt.figure(figsize=(10, 4))
        sns.countplot(x='number_project', data=no_churn_data)
        st.pyplot(plt.gcf())

        # Summary of Non-Churn Data
        st.write("""
        **Summary for Non-Churned Employees**:  
        - Non-churned employees typically have higher satisfaction levels and moderate last evaluation scores.
        - These employees manage a reasonable number of projects and maintain a balanced workload.
        - The XGBoost classifier correctly identifies these patterns to predict employees who are less likely to leave the company.
        """)
