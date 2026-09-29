import streamlit as st
import joblib
import pandas as pd

model = joblib.load('artifacts/burnout_prediction_pipeline.pkl')

def main():
    st.title('Student Burnout Prediction App')
    st.write("Masukkan data mahasiswa di bawah ini untuk memprediksi tingkat burnout (Kelelahan Mental).")

    age = st.number_input("Age", 15, 100, 20)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    course = st.selectbox("Course", ["BTech", "BCA", "BSc", "MBA", "MCA", "BBA"])
    year = st.selectbox("Year of Study", ["1st", "2nd", "3rd", "4th"])
    
    daily_study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 4.0)
    daily_sleep_hours = st.number_input("Daily Sleep Hours", 0.0, 24.0, 7.0)
    screen_time_hours = st.number_input("Screen Time Hours", 0.0, 24.0, 5.0)
    physical_activity_hours = st.number_input("Physical Activity Hours", 0.0, 24.0, 1.0)
    
    stress_level = st.radio("Stress Level", ["Low", "Medium", "High"])
    anxiety_score = st.number_input("Anxiety Score (0-10)", 0, 10, 5)
    depression_score = st.number_input("Depression Score (0-10)", 0, 10, 5)
    academic_pressure_score = st.number_input("Academic Pressure Score (0-10)", 0, 10, 5)
    financial_stress_score = st.number_input("Financial Stress Score (0-10)", 0, 10, 5)
    social_support_score = st.number_input("Social Support Score (0-10)", 0, 10, 5)
    
    sleep_quality = st.radio("Sleep Quality", ["Poor", "Average", "Good"])
    attendance_percentage = st.number_input("Attendance Percentage", 0.0, 100.0, 80.0)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
    internet_quality = st.radio("Internet Quality", ["Poor", "Average", "Good"])
    
    data = {
        'age': int(age), 
        'gender': gender, 
        'course': course, 
        'year': year,
        'daily_study_hours': float(daily_study_hours), 
        'daily_sleep_hours': float(daily_sleep_hours),
        'screen_time_hours': float(screen_time_hours), 
        'stress_level': stress_level,
        'anxiety_score': int(anxiety_score), 
        'depression_score': int(depression_score),
        'academic_pressure_score': int(academic_pressure_score), 
        'financial_stress_score': int(financial_stress_score),
        'social_support_score': int(social_support_score), 
        'physical_activity_hours': float(physical_activity_hours),
        'sleep_quality': sleep_quality, 
        'attendance_percentage': float(attendance_percentage),
        'cgpa': float(cgpa), 
        'internet_quality': internet_quality
    }
    
    cols = ['age', 'gender', 'course', 'year', 'daily_study_hours', 'daily_sleep_hours',
            'screen_time_hours', 'stress_level', 'anxiety_score', 'depression_score',
            'academic_pressure_score', 'financial_stress_score', 'social_support_score',
            'physical_activity_hours', 'sleep_quality', 'attendance_percentage', 'cgpa', 'internet_quality']
            
    df = pd.DataFrame([list(data.values())], columns=cols)

    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        
        if prediction == "High":
            st.error(f"Prediction: {prediction} Burnout Risk")
        elif prediction == "Medium":
            st.warning(f"Prediction: {prediction} Burnout Risk")
        else:
            st.success(f"Prediction: {prediction} Burnout Risk")

if __name__ == "__main__":
    main()