import pandas as pd
import ollama


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """Load a CSV dataset using Pandas."""

    try:
        data = pd.read_csv(file_path)

        print("✅ Dataset loaded successfully!")
        print(f"Rows: {data.shape[0]}")
        print(f"Columns: {data.shape[1]}")

        return data

    except Exception as e:

        print("❌ Error loading dataset:")
        print(e)

        return None


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(data):
    """Analyze basic information about the dataset."""

    print("\n📊 Dataset Analysis")
    print("===================")

    # Dataset shape
    rows = data.shape[0]
    columns = data.shape[1]

    print(f"\nRows: {rows}")
    print(f"Columns: {columns}")

    # Column names
    print("\nColumns:")
    print(data.columns.tolist())

    # Data types
    print("\nData Types:")
    print(data.dtypes)

    # Missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Numerical statistics
    print("\n📈 Numerical Statistics:")
    print(data.describe())

    return {
        "rows": rows,
        "columns": columns,
        "column_names": data.columns.tolist(),
        "data_types": data.dtypes.astype(str).to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "statistics": data.describe().to_dict()
    }


# ============================================================
# ANSWER DATA QUESTIONS
# ============================================================

def answer_question(data, question):
    """Answer simple data-related questions."""

    question = question.lower()

    if "average salary" in question:

        result = data["Salary"].mean()

        return f"The average salary is {result:.2f}"

    elif "average age" in question:

        result = data["Age"].mean()

        return f"The average age is {result:.2f}"

    elif "highest salary" in question:

        result = data["Salary"].max()

        return f"The highest salary is {result}"

    elif "lowest salary" in question:

        result = data["Salary"].min()

        return f"The lowest salary is {result}"

    else:

        return "I cannot answer this question yet."


# ============================================================
# GENERATE PANDAS CODE USING OLLAMA
# ============================================================

def generate_code(question, columns, error_message=None):
    """
    Use Ollama to generate Pandas code.

    If an error occurs, ask Ollama to correct the code.
    """

    if error_message:

        prompt = f"""
You are a Python data analysis assistant.

The dataset has these columns:

{columns}

User question:

{question}

The previously generated code caused this error:

{error_message}

Generate a corrected Python expression using pandas
DataFrame named `data`.

Rules:
- Generate only one Python expression.
- Do not use print().
- Do not use markdown.
- Do not explain anything.
"""

    else:

        prompt = f"""
You are a Python data analysis assistant.

The dataset has these columns:

{columns}

User question:

{question}

Generate only one Python expression using pandas
DataFrame named `data`.

Rules:
- Do not use print().
- Do not use markdown.
- Do not explain anything.

Example:

Question:
What is the average salary?

Answer:
data["Salary"].mean()
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    code = response["message"]["content"].strip()

    return code


# ============================================================
# VALIDATE GENERATED CODE
# ============================================================

def validate_code(code):
    """Validate generated code before execution."""

    forbidden_keywords = [
        "import",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "open(",
        "exec(",
        "eval(",
        "__",
        "socket",
        "requests",
        "urllib"
    ]

    code_lower = code.lower()

    for keyword in forbidden_keywords:

        if keyword in code_lower:
            return False

    if not code.strip().startswith("data"):
        return False

    return True


# ============================================================
# EXECUTE GENERATED CODE
# ============================================================

def execute_code(data, code):
    """Validate and execute generated Pandas expression."""

    if not validate_code(code):

        return "❌ Unsafe code detected. Execution blocked."

    try:

        result = eval(
            code,
            {"__builtins__": {}},
            {"data": data}
        )

        return result

    except Exception as e:

        return f"Error executing code: {e}"


# ============================================================
# SELF-CORRECTION
# ============================================================

def self_correct(data, question, code, error_message):
    """Handle errors from generated Pandas code."""

    available_columns = data.columns.tolist()

    # --------------------------------------------------------
    # Handle common missing-column error
    # --------------------------------------------------------

    if "Error executing code" in error_message:

        error_text = error_message.lower()

        if "'bonus'" in error_text or '"bonus"' in error_text:

            return (
                "❌ The dataset does not contain a 'Bonus' column.\n\n"
                f"Available columns: {', '.join(available_columns)}"
            )

    # --------------------------------------------------------
    # Ask Ollama to correct the generated code
    # --------------------------------------------------------

    corrected_code = generate_code(
        question,
        available_columns,
        error_message
    )

    return corrected_code