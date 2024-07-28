import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
stud_data = pd.read_csv('Binary/BinarySelectedFeatures.csv')

# Fit the encoders based on the unique values in the dataset columns
highest_education_encoder = LabelEncoder()
highest_education_encoder.fit(stud_data['highest_education'])

imd_band_encoder = LabelEncoder()
imd_band_encoder.fit(stud_data['imd_band'])

# Save the label encoders
joblib.dump(highest_education_encoder, 'Deployment/highest_education_encoder.pkl')
joblib.dump(imd_band_encoder, 'Deployment/imd_band_encoder.pkl')

# Exclude 'at_risk_binary_encoded' when fitting the scaler
numeric_columns = stud_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns.remove('at_risk_binary_encoded')

# Fit and save the scaler
scaler = StandardScaler()
scaler.fit(stud_data[numeric_columns])
joblib.dump(scaler, 'Deployment/scaler.pkl')
