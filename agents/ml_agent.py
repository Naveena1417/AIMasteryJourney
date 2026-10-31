import pandas as pd
import ollama
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("http://mlflow-server:5000")

mlflow.set_experiment("AI_Research_ML_Crew")


# ============================================================
# 1. LOAD ML DATASET
# ============================================================

def load_ml_dataset(file_path):
    """
    Load ML dataset using Pandas.
    """

    try:

        data = pd.read_csv(file_path)

        print("✅ ML Dataset loaded successfully!")
        print(f"Rows: {data.shape[0]}")
        print(f"Columns: {data.shape[1]}")

        return data

    except Exception as e:

        print("❌ Error loading ML dataset:")
        print(e)

        return None


# ============================================================
# 2. TRAIN CLASSIFICATION MODEL
# ============================================================

def train_classification_model(
    data,
    target_column,
    model_name="logistic_regression"
):
    """
    Train a classification model.

    Supported models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    """

    # ========================================================
    # 1. FEATURES AND TARGET
    # ========================================================

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    # ========================================================
    # 2. TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n📊 Dataset Split")
    print("================")
    print(
        f"Training samples: {len(X_train)}"
    )
    print(
        f"Testing samples: {len(X_test)}"
    )

    # ========================================================
    # 3. SELECT MODEL
    # ========================================================

    if model_name == "logistic_regression":

        model = LogisticRegression(
            max_iter=1000
        )

    elif model_name == "random_forest":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    elif model_name == "xgboost":

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # ========================================================
    # 4. MLFLOW EXPERIMENT RUN
    # ========================================================

    with mlflow.start_run():

        # ----------------------------------------------------
        # Log Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "task",
            "classification"
        )

        mlflow.log_param(
            "target_column",
            target_column
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "testing_samples",
            len(X_test)
        )

        # ----------------------------------------------------
        # Train Model
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log Metric
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )
        # ----------------------------------------------------
        # Log Trained Model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="trained_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )
        # ----------------------------------------------------
        # Print Results
        # ----------------------------------------------------

        print("\n🤖 Model Training Completed")
        print("===========================")

        print(
            f"Model: {model_name}"
        )

        print(
            f"Accuracy: {accuracy:.2f}"
        )

        print("\n📋 Classification Report")
        print("========================")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

    # ========================================================
    # RETURN MODEL AND ACCURACY
    # ========================================================

    return model, accuracy


# ============================================================
# 3. LLM ANALYSIS
# ============================================================

def analyze_ml_request_with_llm(
    question,
    columns
):
    """
    Use Ollama LLM to identify:
    1. ML task
    2. Target column
    3. Best model
    """

    prompt = f"""
You are an ML assistant.

Dataset columns:
{columns}

User question:
{question}

Identify the following:

1. task
2. target_column
3. model

Task must be:
classification

Target column must be one of the dataset columns.

Model must be one of:
logistic_regression
random_forest
xgboost

Return ONLY valid JSON.

Example:
{{
    "task": "classification",
    "target_column": "Purchased",
    "model": "xgboost"
}}

Do not add explanation.
Do not use markdown.
"""

    # ========================================================
    # CALL OLLAMA
    # ========================================================

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # ========================================================
    # GET LLM RESPONSE
    # ========================================================

    result = response[
        "message"
    ][
        "content"
    ].strip()

    print("\n🤖 LLM Analysis")
    print("================")
    print(result)

    # ========================================================
    # PARSE JSON
    # ========================================================

    import json

    try:

        analysis = json.loads(
            result
        )

        # ====================================================
        # VALIDATE MODEL
        # ====================================================

        allowed_models = [
            "logistic_regression",
            "random_forest",
            "xgboost"
        ]

        if analysis["task"] != "classification":

            raise ValueError(
                "Unsupported ML task"
            )

        # ====================================================
        # VALIDATE TARGET COLUMN
        # ====================================================

        if analysis["target_column"] not in columns:

            raise ValueError(
                "Invalid target column"
            )

        # ====================================================
        # VALIDATE MODEL
        # ====================================================

        if analysis["model"] not in allowed_models:

            raise ValueError(
                "Invalid model"
            )

        return analysis

    except Exception as e:

        print("❌ LLM analysis error:")
        print(e)

        # ====================================================
        # FALLBACK
        # ====================================================

        return {
            "task": "classification",
            "target_column": None,
            "model": "logistic_regression"
        }