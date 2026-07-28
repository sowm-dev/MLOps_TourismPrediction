import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Enter the sensor and configuration data below to get a prediction.
""")

Age         = st.number_input(label="Age",min_value=18,max_value=61,value=18,step=1)
CityTier     = st.number_input(label="CityTier",min_value=1,max_value=3,value=1,step=1)
NumberOfPersonVisiting = st.number_input(label="NumberOfPersonVisiting",min_value=1,max_value=5,value=1,step=1)
PreferredPropertyStar    = st.number_input(label="PreferredPropertyStar",min_value=3,max_value=5,value=3,step=1)
NumberOfTrips       = st.number_input(label="NumberOfTrips",min_value=1,max_value=22,value=1,step=1)
Passport    = st.number_input(label="Passport",min_value=0,max_value=1,value=1,step=1)

OwnCar         = st.number_input(label="OwnCar",min_value=0,max_value=1,value=1,step=1)
NumberOfChildrenVisiting     = st.number_input(label="NumberOfChildrenVisiting",min_value=0,max_value=3,value=1,step=1)
NumberOfPersonVisiting = st.number_input(label="NumberOfPersonVisiting",min_value=1,max_value=5,value=1,step=1)
MonthlyIncome    = st.number_input(label="MonthlyIncome",min_value=1000,max_value=89678,value=1,step=1)
PitchSatisfactionScore       = st.number_input(label="PitchSatisfactionScore",min_value=1,max_value=5,value=1,step=1)
NumberOfFollowups    = st.number_input(label="NumberOfFollowups",min_value=1,max_value=6,value=1,step=1)

DurationOfPitch         = st.number_input(label="DurationOfPitch",min_value=5,max_value=127,value=5,step=1)
TypeofContact     = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business","Large Business"])
Gender    = st.selectbox("Gender", ["Female", "Male"])
ProductPitched       = st.selectbox("ProductPitched", ["Deluxe", "Basic", "Standard","Super Deluxe","King"])
MaritalStatus    = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced","Unmarried"])
Designation    = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager","AVP","VP"])



input_data = pd.DataFrame([{
    "Age": Age,
    "City Tier": CityTier,
    "Number Of PersonVisiting": NumberOfPersonVisiting,
    "Preferred Property Star": PreferredPropertyStar,
    "Number Of Trips": NumberOfTrips,
    "Passport": Passport,
        "OwnCar": OwnCar,
    "Number Of Children Visiting": NumberOfChildrenVisiting,
    "Number Of Person Visiting": NumberOfPersonVisiting,
    "Monthly Income": MonthlyIncome,
    "Pitch Satisfaction Score": PitchSatisfactionScore,
    "Number Of Followups": NumberOfFollowups,
    "Duration Of Pitch": DurationOfPitch,
            "Type of Contact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "Product Pitched": ProductPitched,
    "Marital Status": MaritalStatus,
    "Designation": Designation,
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
