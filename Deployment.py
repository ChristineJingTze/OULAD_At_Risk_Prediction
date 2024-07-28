import streamlit as st
import pandas as pd
import joblib
import base64

# Load the model, encoders, and scaler
xgb_model = joblib.load('Binary/Model/Tuned/best_xgb_model_binary.pkl')
highest_education_encoder = joblib.load('Deployment/highest_education_encoder.pkl')
imd_band_encoder = joblib.load('Deployment/imd_band_encoder.pkl')
scaler = joblib.load('Deployment/scaler.pkl')

# Numeric features excluding the target variable and sum_click for user input
numeric_features = [
    'studied_credits',
    'num_of_prev_attempts',
    'before_course_click',
    'after_course_click',
    'final_score'
]

# Thresholds for each feature
thresholds = {
    'num_of_prev_attempts': 0,
    'studied_credits': 30,
    'before_course_click': 10,
    'after_course_click': 50,
    'final_score': 40
}

# Function to preprocess user input
def preprocess_input(highest_education, imd_band, numeric_values):
    highest_education_encoded = highest_education_encoder.transform([highest_education])[0]
    imd_band_encoded = imd_band_encoder.transform([imd_band])[0]
    
    # Calculate sum_click
    before_course_click = numeric_values[2]
    after_course_click = numeric_values[3]
    sum_click = before_course_click + after_course_click
    
    # Create a list with the proper order of features
    features = [highest_education_encoded, imd_band_encoded] + numeric_values[:2] + [sum_click] + numeric_values[2:]

    scaled_features = scaler.transform([features])
    return scaled_features

# Function to preprocess CSV data
def preprocess_csv(input_data):
    input_data['highest_education_encoded'] = highest_education_encoder.transform(input_data['highest_education'])
    input_data['imd_band_encoded'] = imd_band_encoder.transform(input_data['imd_band'])
    
    # Calculate sum_click
    input_data['sum_click'] = input_data['before_course_click'] + input_data['after_course_click']
    
    # Ensure the columns are in the same order as they were fitted
    input_data_selected = input_data[['studied_credits', 'num_of_prev_attempts','imd_band_encoded', 'highest_education_encoded','before_course_click', 'sum_click', 'after_course_click', 'final_score']]
    
    input_data_scaled = scaler.transform(input_data_selected)
    return input_data_scaled

# Function to interpret the prediction
def interpret_prediction(prediction):
    return "At Risk" if prediction >= 0.5 else "Not At Risk"

# Function to decode the encoded features
def decode_features(encoded_df):
    decoded_df = encoded_df.copy()
    decoded_df['highest_education'] = highest_education_encoder.inverse_transform(encoded_df['highest_education_encoded'])
    decoded_df['imd_band'] = imd_band_encoder.inverse_transform(encoded_df['imd_band_encoded'])
    return decoded_df

# Function to provide advice based on feature thresholds
def provide_feature_based_advice(numeric_values, thresholds):
    advice = []
    high_risk_factors = []

    if numeric_values[0] > thresholds['num_of_prev_attempts']:
        advice.append("Consider providing additional support and resources to reduce the number of previous attempts.")
        high_risk_factors.append("Number of Previous Attempts")
    if numeric_values[1] < thresholds['studied_credits']:
        advice.append("Review the student's studied credits and offer guidance on managing their course load effectively.")
        high_risk_factors.append("Studied Credits")
    if numeric_values[2] < thresholds['before_course_click']:
        advice.append("Encourage increased engagement with preparatory materials before the course begins.")
        high_risk_factors.append("Before Course Clicks")
    if numeric_values[3] < thresholds['after_course_click']:
        advice.append("Promote consistent engagement throughout the course to improve understanding and performance.")
        high_risk_factors.append("After Course Clicks")
    if numeric_values[4] < thresholds['final_score']:
        advice.append("Focus on improving final scores through better study habits and offering additional help if needed.")
        high_risk_factors.append("Final Score")
    
    # Check if all features do not meet the thresholds
    if not advice:
        advice.append("Approach the students to gather more information and understand their challenges better.")
    
    return advice, high_risk_factors

def format_advice_for_csv(advice_list):
    return " ".join([advice + "\n" for advice in advice_list])

def generate_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encode
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# Streamlit app
st.title("🎯 Prediction on At-Risk Students")
st.image('Deployment/image.jpeg', use_column_width=True)

# Inject CSS for text wrapping and row height
st.markdown("""
<style>
.dataframe td {
    white-space: wrap;
    word-wrap: break-word;
    height: auto;
}
.dataframe td.col-Advice\ for\ Students {
    height: 100px;  /* Adjust height as needed */
}
</style>
""", unsafe_allow_html=True)

# Option to choose between user input or CSV upload
option = st.selectbox("Choose input method:", ("User Input", "CSV Upload"))

if option == "User Input":
    st.header("User Input")
    st.info("Please enter the student's details to get the risk prediction.")
    
    highest_education = st.selectbox("Highest Education", highest_education_encoder.classes_, help="Select the highest level of education attained by the student.")
    imd_band = st.selectbox("IMD Band", imd_band_encoder.classes_, help="Select the IMD band representing the student's socio-economic status.")
    
    numeric_values = []
    for feature in numeric_features:
        if feature == 'num_of_prev_attempts':
            label = "Num of Previous Attempts"
            step_value = 1
        elif feature == 'studied_credits':
            label = "Studied Credits"
            step_value = 10
        elif feature == 'final_score':
            label = "Final Score"
            step_value = 0.1
        else:
            label = " ".join([word.capitalize() for word in feature.split('_')])
            step_value = 1
        
        if feature == 'final_score':
            value = st.number_input(f"{label}", value=0.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f", help="Enter the student's final score. It should be a value between 0 and 100 with one decimal place.")
            if value > 100.0:
                st.error("Final Score cannot be more than 100.")
                continue
        else:
            value = st.number_input(f"{label}", value=0, min_value=0, step=step_value, help=f"Enter the student's {label.lower()}.")
        numeric_values.append(value)
    
    # Calculate sum_click
    sum_click = numeric_values[2] + numeric_values[3]
    st.write(f"Sum Click: {sum_click}")
    
    if st.button("Predict"):
        user_data = preprocess_input(highest_education, imd_band, numeric_values)
        prediction = xgb_model.predict(user_data)[0]
        probability = xgb_model.predict_proba(user_data)[0][1]
        result = interpret_prediction(prediction)
        st.write(f"Prediction: {result} (Probability: {probability:.2f})")

        if result == "At Risk":
            st.error(f"The student is {result} with a probability of {probability:.2f}")
            st.subheader("Advice")
            advice, high_risk_factors = provide_feature_based_advice(numeric_values, thresholds)
            for tip in advice:
                st.write(f"- {tip}")
            
            st.write("### High Risk Factors")
            for factor in high_risk_factors:
                st.write(f"- {factor}")
        else:
            st.success(f"The student is {result} with a probability of {probability:.2f}")


elif option == "CSV Upload":
    st.header("CSV Upload")
    st.info("Upload a CSV file containing the student data to get the risk predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        # Ensure necessary columns are present
        required_columns = [
            'highest_education', 'imd_band',
            'num_of_prev_attempts', 'studied_credits',
            'before_course_click', 'after_course_click', 'final_score'
        ]
        
        if not set(required_columns).issubset(input_data.columns):
            st.error("CSV file must contain all necessary columns.")
        else:
            # Ensure no negative values are present in the numeric columns
            if (input_data[numeric_features] < 0).any().any():
                st.error("Numeric values cannot be negative.")
            else:
                input_data_scaled = preprocess_csv(input_data)
                
                predictions = xgb_model.predict(input_data_scaled)
                probabilities = xgb_model.predict_proba(input_data_scaled)[:, 1]
                input_data['Prediction'] = ["At Risk" if pred >= 0.5 else "Not At Risk" for pred in predictions]
                input_data['Probability'] = probabilities
                
                # Decode the encoded features
                decoded_input_data = decode_features(input_data)
                
                # Provide feature-based advice for each student if they are at risk
                input_data['Advice for Students'] = input_data.apply(
                    lambda row: format_advice_for_csv(
                        provide_feature_based_advice(
                            [row['num_of_prev_attempts'], row['studied_credits'], row['before_course_click'], row['after_course_click'], row['final_score']],
                            thresholds
                        )[0]
                    ) if row['Prediction'] == "At Risk" else "", axis=1
                )
                
                input_data['High Risk Factors'] = input_data.apply(
                    lambda row: ", ".join(
                        provide_feature_based_advice(
                            [row['num_of_prev_attempts'], row['studied_credits'], row['before_course_click'], row['after_course_click'], row['final_score']],
                            thresholds
                        )[1]
                    ) if row['Prediction'] == "At Risk" else "", axis=1
                )
                
                # Merge the advice and high risk factors columns back to decoded_input_data
                decoded_input_data['Advice for Students'] = input_data['Advice for Students']
                decoded_input_data['High Risk Factors'] = input_data['High Risk Factors']
                
                # Rename the columns for better readability
                decoded_input_data = decoded_input_data.rename(columns={
                    'highest_education': 'Highest Education',
                    'imd_band': 'IMD Band',
                    'num_of_prev_attempts': 'Number of Previous Attempts',
                    'studied_credits': 'Studied Credits',
                    'before_course_click': 'Before Course Clicks',
                    'after_course_click': 'After Course Clicks',
                    'final_score': 'Final Score',
                    'Prediction': 'Risk Prediction',
                    'Probability': 'Risk Probability'
                })
                
                # Drop the encoded columns
                decoded_input_data = decoded_input_data.loc[:, ~decoded_input_data.columns.str.contains('_encoded')]
                
                # Sidebar filters
                st.sidebar.header("Filter Data")

                # Highest Education filter
                filter_highest_education = st.sidebar.multiselect("Highest Education", options=decoded_input_data["Highest Education"].unique(), default=decoded_input_data["Highest Education"].unique())

                # IMD Band filter
                filter_imd_band = st.sidebar.multiselect("IMD Band", options=decoded_input_data["IMD Band"].unique(), default=decoded_input_data["IMD Band"].unique())
                
                filter_num_of_prev_attempts = st.sidebar.slider("Number of Previous Attempts", min_value=int(decoded_input_data["Number of Previous Attempts"].min()), max_value=int(decoded_input_data["Number of Previous Attempts"].max()), value=(int(decoded_input_data["Number of Previous Attempts"].min()), int(decoded_input_data["Number of Previous Attempts"].max())))
                filter_studied_credits = st.sidebar.slider("Studied Credits", min_value=int(decoded_input_data["Studied Credits"].min()), max_value=int(decoded_input_data["Studied Credits"].max()), value=(int(decoded_input_data["Studied Credits"].min()), int(decoded_input_data["Studied Credits"].max())))
                filter_before_course_click = st.sidebar.slider("Before Course Clicks", min_value=int(decoded_input_data["Before Course Clicks"].min()), max_value=int(decoded_input_data["Before Course Clicks"].max()), value=(int(decoded_input_data["Before Course Clicks"].min()), int(decoded_input_data["Before Course Clicks"].max())))
                filter_after_course_click = st.sidebar.slider("After Course Clicks", min_value=int(decoded_input_data["After Course Clicks"].min()), max_value=int(decoded_input_data["After Course Clicks"].max()), value=(int(decoded_input_data["After Course Clicks"].min()), int(decoded_input_data["After Course Clicks"].max())))
                filter_final_score = st.sidebar.slider("Final Score", min_value=float(decoded_input_data["Final Score"].min()), max_value=float(decoded_input_data["Final Score"].max()), value=(float(decoded_input_data["Final Score"].min()), float(decoded_input_data["Final Score"].max())))

                # Apply filters dynamically
                filtered_data = decoded_input_data[
                    (decoded_input_data["Highest Education"].isin(filter_highest_education)) &
                    (decoded_input_data["IMD Band"].isin(filter_imd_band)) &
                    (decoded_input_data["Number of Previous Attempts"] >= filter_num_of_prev_attempts[0]) &
                    (decoded_input_data["Number of Previous Attempts"] <= filter_num_of_prev_attempts[1]) &
                    (decoded_input_data["Studied Credits"] >= filter_studied_credits[0]) &
                    (decoded_input_data["Studied Credits"] <= filter_studied_credits[1]) &
                    (decoded_input_data["Before Course Clicks"] >= filter_before_course_click[0]) &
                    (decoded_input_data["Before Course Clicks"] <= filter_before_course_click[1]) &
                    (decoded_input_data["After Course Clicks"] >= filter_after_course_click[0]) &
                    (decoded_input_data["After Course Clicks"] <= filter_after_course_click[1]) &
                    (decoded_input_data["Final Score"] >= filter_final_score[0]) &
                    (decoded_input_data["Final Score"] <= filter_final_score[1])
                ]

                # Display the filtered dataset
                st.subheader("Filtered Data")
                st.dataframe(filtered_data, width=1000, height=500)

                # Provide a download link for the filtered data
                st.markdown(generate_csv_download_link(filtered_data, "filtered_data.csv"), unsafe_allow_html=True)
