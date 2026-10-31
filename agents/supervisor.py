import ollama
import os


MODEL_NAME = "qwen2.5:3b"


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def route_question(question):
    """
    Supervisor Agent routes the user's question
    to research, data, or ml agent.
    """

    prompt = f"""
You are a strict supervisor AI agent.

Your job is to route the user's question to exactly ONE agent.

Available agents:

============================================================
RESEARCH AGENT
============================================================

Use research for:

- AI/ML theory
- Machine learning concepts
- Definitions
- Explanations
- Knowledge questions
- Questions about documents
- RAG-related knowledge questions
- NLP concepts
- Deep learning concepts
- Transformer concepts

Examples:

"What is supervised learning?"
"What is deep learning?"
"What is a transformer?"
"Explain overfitting."
"What is RAG?"


============================================================
DATA AGENT
============================================================

Use data for:

- Questions about the CSV dataset
- Dataset values
- Salary
- Age
- Employee information
- Average
- Maximum
- Minimum
- Count
- Sum
- Filtering
- Sorting
- Grouping
- Data analysis

Examples:

"What is the highest salary?"
"What is the average age?"
"How many employees are there?"
"Who has the highest salary?"
"What is the minimum salary?"


============================================================
ML AGENT
============================================================

Use ml for:

- Model training
- Prediction
- Classification
- Regression
- Model evaluation
- Model selection
- Feature engineering
- Training machine learning models
- Making predictions

Examples:

"Train a classification model."
"Predict whether a customer will purchase."
"Which model should I use?"
"Build a regression model."
"Evaluate the model accuracy."


============================================================
IMPORTANT ROUTING RULES
============================================================

1. If the question is about DATASET VALUES or DATA ANALYSIS,
   choose DATA.

2. If the question asks for a MAXIMUM, MINIMUM, AVERAGE,
   COUNT, SUM, FILTER, SORT, or GROUP operation on the dataset,
   choose DATA.

3. If the question asks to TRAIN, PREDICT, CLASSIFY,
   REGRESS, EVALUATE, or SELECT A MODEL,
   choose ML.

4. General AI/ML theory and definitions should go to RESEARCH.

5. The phrase "what is" does NOT automatically mean RESEARCH.

Example:

"What is supervised learning?"
→ research

"What is the highest salary?"
→ data

"What is the average age?"
→ data

"What is the accuracy of the model?"
→ ml


User Question:
{question}

Return ONLY one word:

research
OR
data
OR
ml
"""
    client = ollama.Client(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    result = response["message"]["content"].strip().lower()

    print("\n🧭 Supervisor Raw Response")
    print("==========================")
    print(result)

    # ========================================================
    # Exact matching
    # ========================================================

    if result == "research":
        return "research"

    if result == "data":
        return "data"

    if result == "ml":
        return "ml"

    # ========================================================
    # Keyword fallback
    # ========================================================

    question_lower = question.lower()

    # --------------------------------------------------------
    # DATA KEYWORDS
    # --------------------------------------------------------

    data_keywords = [
        "salary",
        "age",
        "highest salary",
        "lowest salary",
        "maximum",
        "minimum",
        "average",
        "mean",
        "median",
        "count",
        "how many",
        "sum",
        "total",
        "dataset",
        "employee",
        "employees",
        "csv",
        "rows",
        "columns",
        "filter",
        "sort",
        "group by"
    ]

    # --------------------------------------------------------
    # ML KEYWORDS
    # --------------------------------------------------------

    ml_keywords = [
        "predict",
        "prediction",
        "train model",
        "training model",
        "classification",
        "classification model",
        "regression",
        "regression model",
        "model selection",
        "machine learning model",
        "feature engineering",
        "evaluate model",
        "model accuracy",
        "confusion matrix",
        "f1 score",
        "precision",
        "recall"
    ]

    # --------------------------------------------------------
    # RESEARCH KEYWORDS
    # --------------------------------------------------------

    research_keywords = [
        "what is",
        "what are",
        "explain",
        "define",
        "meaning",
        "concept",
        "how does",
        "difference between",
        "why",
        "rag",
        "transformer",
        "neural network",
        "deep learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "nlp",
        "llm",
        "large language model"
    ]

    # ========================================================
    # PRIORITY ORDER
    # ========================================================
    #
    # DATA → ML → RESEARCH
    #
    # This prevents:
    #
    # "What is the highest salary?"
    #
    # from being incorrectly routed to research
    # because of the phrase "what is".
    # ========================================================

    if any(
        keyword in question_lower
        for keyword in data_keywords
    ):
        return "data"

    if any(
        keyword in question_lower
        for keyword in ml_keywords
    ):
        return "ml"

    if any(
        keyword in question_lower
        for keyword in research_keywords
    ):
        return "research"

    # ========================================================
    # DEFAULT
    # ========================================================

    return "research"