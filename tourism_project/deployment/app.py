import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")
st.write("""
Predict whether a customer is likely to purchase the Wellness Tourism Package based on demographic and interaction details.
""")

# Age         = st.number_input(label="Age",min_value=18,max_value=61,value=18,step=1)
# CityTier     = st.number_input(label="CityTier",min_value=1,max_value=3,value=1,step=1)
# NumberOfPersonVisiting = st.number_input(label="NumberOfPersonVisiting",min_value=1,max_value=5,value=1,step=1)
# PreferredPropertyStar    = st.number_input(label="PreferredPropertyStar",min_value=3,max_value=5,value=3,step=1)
# NumberOfTrips       = st.number_input(label="NumberOfTrips",min_value=1,max_value=22,value=1,step=1)
# Passport    = st.number_input(label="Passport",min_value=0,max_value=1,value=1,step=1)
# OwnCar         = st.number_input(label="OwnCar",min_value=0,max_value=1,value=1,step=1)
# NumberOfChildrenVisiting     = st.number_input(label="NumberOfChildrenVisiting",min_value=0,max_value=3,value=1,step=1)
# MonthlyIncome    = st.number_input(label="MonthlyIncome",min_value=1000,max_value=89678,value=1,step=1)
# PitchSatisfactionScore       = st.number_input(label="PitchSatisfactionScore",min_value=1,max_value=5,value=1,step=1)
# NumberOfFollowups    = st.number_input(label="NumberOfFollowups",min_value=1,max_value=6,value=1,step=1)
# DurationOfPitch         = st.number_input(label="DurationOfPitch",min_value=5,max_value=127,value=5,step=1)
# TypeofContact     = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
# Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business","Large Business"])
# Gender    = st.selectbox("Gender", ["Female", "Male"])
# ProductPitched       = st.selectbox("ProductPitched", ["Deluxe", "Basic", "Standard","Super Deluxe","King"])
# MaritalStatus    = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced","Unmarried"])
# Designation    = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager","AVP","VP"])


# -------------------------
# Numerical Inputs
# -------------------------

Age = st.number_input("Age", min_value=18, max_value=61, value=30)

CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=1)

NumberOfPersonVisiting = st.number_input(
    "NumberOfPersonVisiting",
    min_value=1,
    max_value=5,
    value=2
)

PreferredPropertyStar = st.number_input(
    "PreferredPropertyStar",
    min_value=3,
    max_value=5,
    value=3
)

NumberOfTrips = st.number_input(
    "NumberOfTrips",
    min_value=1,
    max_value=22,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "OwnCar",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "NumberOfChildrenVisiting",
    min_value=0,
    max_value=3,
    value=0
)

MonthlyIncome = st.number_input(
    "MonthlyIncome",
    min_value=1000,
    max_value=90000,
    value=25000
)

PitchSatisfactionScore = st.number_input(
    "PitchSatisfactionScore",
    min_value=1,
    max_value=5,
    value=3
)

NumberOfFollowups = st.number_input(
    "NumberOfFollowups",
    min_value=1,
    max_value=6,
    value=2
)

DurationOfPitch = st.number_input(
    "DurationOfPitch",
    min_value=5,
    max_value=127,
    value=20
)

# -------------------------
# Categorical Inputs
# -------------------------

TypeofContact = st.selectbox(
    "TypeofContact",
    ["Self Enquiry", "Company Invited"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

ProductPitched = st.selectbox(
    "ProductPitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

MaritalStatus = st.selectbox(
    "MaritalStatus",
    ["Single", "Married", "Divorced", "Unmarried"]
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# -------------------------------------------------
# Create Input DataFrame
# -------------------------------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])


# -------------------------------------------------
# Feature Engineering
# (Must match training pipeline)
# -------------------------------------------------

input_data["FamilySize"] = (
    input_data["NumberOfPersonVisiting"] +
    input_data["NumberOfChildrenVisiting"]
)

input_data["IncomePerPerson"] = (
    input_data["MonthlyIncome"] /
    (input_data["NumberOfPersonVisiting"] + 1)
)

input_data["FrequentTraveller"] = (
    input_data["NumberOfTrips"] >= 4
).astype(int)

input_data["InternationalTraveller"] = (
    (input_data["Passport"] == 1) &
    (input_data["NumberOfTrips"] >= 2)
).astype(int)


# input_data = pd.DataFrame([{
#     "Age": Age,
#     "City Tier": CityTier,
#     "Number Of PersonVisiting": NumberOfPersonVisiting,
#     "Preferred Property Star": PreferredPropertyStar,
#     "Number Of Trips": NumberOfTrips,
#     "Passport": Passport,
#     "OwnCar": OwnCar,
#     "Number Of Children Visiting": NumberOfChildrenVisiting,
#     "Monthly Income": MonthlyIncome,
#     "Pitch Satisfaction Score": PitchSatisfactionScore,
#     "Number Of Followups": NumberOfFollowups,
#     "Duration Of Pitch": DurationOfPitch,
#     "Type of Contact": TypeofContact,
#     "Occupation": Occupation,
#     "Gender": Gender,
#     "Product Pitched": ProductPitched,
#     "Marital Status": MaritalStatus,
#     "Designation": Designation,
# }])

# if st.button("Predict Package purchase"):
#     prediction = model.predict(input_data)[0]
#     result = "Package purchased" if prediction == 1 else "No Failure"
#     st.subheader("Prediction Result:")
#     st.success(f"The model predicts: **{result}**")


 # -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.error(
            "Customer is unlikely to purchase the Wellness Tourism Package."
        )   
