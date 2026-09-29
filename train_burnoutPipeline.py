import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def train_model(x_train, y_train):
    cat_feat = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Pipeline Preprocessing
    numeric_preprocess = Pipeline([('num_imputer', SimpleImputer(strategy='mean'))])
    
    categorical_preprocess= Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent')),
                                      ('cat_encoder', OrdinalEncoder(categories=[
                                          ['Male', 'Female', 'Other'],                   
                                          ['BTech', 'BCA', 'BSc', 'MBA', 'MCA', 'BBA'],  
                                          ['Low', 'Medium', 'High'],                     
                                          ['Poor', 'Average', 'Good'],                   
                                          ['Poor', 'Average', 'Good']                    
                                      ]))])
    
    preprocess = ColumnTransformer(transformers=[
        ('numPreprocess', numeric_preprocess, num_feat),
        ('catPreprocess', categorical_preprocess, (['gender', 'course', 'stress_level', 'sleep_quality', 'internet_quality']))
        ], remainder='drop')

    burnout_pred = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', RandomForestClassifier(criterion='gini', max_depth=4))
    ])
                                  
    mlflow.set_experiment("Student Burnout Prediction")

    with mlflow.start_run() as run:
        # log parameters
        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 4)

        # train model
        burnout_pred.fit(x_train, y_train)

        joblib.dump(burnout_pred, ARTIFACTS_DIR / "burnout_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(burnout_pred, artifact_path="model")

    return run.info.run_id