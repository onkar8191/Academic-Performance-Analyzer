import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("Academic Performance Analyzer")
st.write("ML based Student Pass/Fail Prediction System")

data = pd.read_csv("student_data.csv")

st.subheader("Student Dataset")
st.dataframe(data)

X = data[["study_hours", "attendance", "previous_marks", "assignment_score"]]
y = data["result"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.subheader("Enter Student Details")

study_hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_marks = st.slider("Previous Marks", 0, 100, 60)
assignment_score = st.slider("Assignment Score", 0, 100, 65)

if st.button("Predict Performance"):
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_marks, assignment_score]],
        columns=["study_hours", "attendance", "previous_marks", "assignment_score"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == "Pass":
        st.success("Prediction: Student will PASS")
    else:
        st.error("Prediction: Student will FAIL")

st.subheader("Performance Graph")

fig, ax = plt.subplots()
ax.bar(
    ["Study Hours", "Attendance", "Previous Marks", "Assignment"],
    [study_hours, attendance, previous_marks, assignment_score]
)
ax.set_ylabel("Score")
st.pyplot(fig)

st.subheader("Algorithm Used")
st.write("Random Forest Classifier")
