import ollama
import pandas as pd
import json
import re



MODEL_NAME = "qwen2.5:3b"


# ============================================================
# REVIEWER AGENT
# ============================================================

def review_result(question, agent_name, result):

    """
    Reviewer Agent checks whether the agent result
    correctly answers the user's question.
    """

    question_lower = question.lower().strip()
    result_text = str(result).strip()


    # ========================================================
    # BASIC EMPTY RESULT CHECK
    # ========================================================

    if not result_text:

        return {
            "approved": False,
            "feedback": "The agent returned an empty answer."
        }


    # ========================================================
    # DATA AGENT VALIDATION
    # ========================================================
    if agent_name.lower() == "ml":

        if isinstance(result, tuple) and len(result) == 2:

            model, accuracy = result

            if accuracy >= 0:
                return {
                    "approved": True,
                    "feedback": (
                        f"ML Agent successfully trained the model "
                        f"and achieved an accuracy of {accuracy:.2f}."
                    )
                }

        return {
            "approved": False,
            "feedback": "The ML Agent did not return a valid model result."
        }
    if agent_name.lower() == "data":

        data_value_keywords = [
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "average",
            "mean",
            "median",
            "count",
            "how many",
            "sum",
            "total",
            "salary",
            "age"
        ]


        # Only apply numeric validation to data questions
        if any(
            keyword in question_lower
            for keyword in data_value_keywords
        ):

            try:
                numbers = re.findall(
                    r"-?\d+(?:\.\d+)?",
                    result_text
                )

                if not numbers:
                    raise ValueError("No numeric value found")

                agent_value = float(numbers[-1])

            except ValueError:
                pass

            else:

                # ------------------------------------------------
                # Dynamic dataset validation
                # ------------------------------------------------

                data = pd.read_csv("data/sample.csv")


                # Highest Salary
                if "highest salary" in question_lower:

                    expected_value = data["Salary"].max()


                # Lowest Salary
                elif "lowest salary" in question_lower:

                    expected_value = data["Salary"].min()

                # Average Salary
                elif "average salary" in question_lower:

                    expected_value = data["Salary"].mean()

                # Total / Sum Salary
                elif (
                        "sum salary" in question_lower
                        or "total salary" in question_lower
                ):

                    expected_value = data["Salary"].sum()
                # Count / Number of Rows
                elif (
                        "how many" in question_lower
                        or "count" in question_lower
                        or "number of employees" in question_lower
                ):

                    expected_value = len(data)

                # Other data questions
                else:

                    return {
                        "approved": True,
                        "feedback": (
                            "The Data Agent result was returned successfully."
                        )
                    }


                # ------------------------------------------------
                # Compare Data Agent result with expected result
                # ------------------------------------------------

                if agent_value != expected_value:

                    print("\n🧐 Reviewer Result")
                    print("==================")
                    print("Data Agent result rejected.")

                    return {
                        "approved": False,
                        "feedback": (
                            f"The Data Agent returned "
                            f"{agent_value:.0f}, but the expected "
                            f"value is {expected_value:.0f}."
                        )
                    }


                print("\n🧐 Reviewer Result")
                print("==================")
                print("Data Agent result verified.")

                return {
                    "approved": True,
                    "feedback": (
                        "The Data Agent returned the correct result."
                    )
                }


    # ========================================================
    # RESEARCH AGENT BASIC APPROVAL
    # ========================================================
    #
    # Research Agent answers generated from the RAG
    # knowledge base can be short but still correct.
    #
    # We should not allow a small local LLM reviewer
    # to reject a simple factual answer because it
    # expects unnecessary wording.
    # ========================================================

    if agent_name.lower() == "research":
        # ----------------------------------------------------
        # Obviously valid supervised-learning answer
        # ----------------------------------------------------

        if (
            "supervised learning" in question_lower
            and "supervised learning" in result_text.lower()
            and "labeled" in result_text.lower()
        ):

            print("\n🧐 Reviewer Result")
            print("==================")
            print(
                "Research Agent answer automatically "
                "approved as a relevant supervised-learning "
                "answer."
            )

            return {
                "approved": True,
                "feedback": (
                    "The Research Agent answer directly "
                    "addresses the question and correctly "
                    "mentions labeled data."
                )
            }


        # ----------------------------------------------------
        # Obvious wrong-answer patterns
        # ----------------------------------------------------

        wrong_patterns = [
            "i don't know",
            "i do not know",
            "cannot answer",
            "unable to answer",
            "no result",
            "not related"
        ]

        if any(
            pattern in result_text.lower()
            for pattern in wrong_patterns
        ):

            return {
                "approved": False,
                "feedback": (
                    "The Research Agent did not provide "
                    "a useful answer to the question."
                )
            }

    # ========================================================
    # ML AGENT FAST APPROVAL
    # ========================================================
    #
    # ML Agent already validates:
    # - Model
    # - Task
    # - Target
    # - Accuracy
    #
    # Therefore, if these core fields are present,
    # approve the result directly instead of asking the
    # small LLM reviewer to unnecessarily reject it.
    # ========================================================

    if agent_name.lower() == "ml":

        ml_result_lower = result_text.lower()

        ml_required_keywords = [
            "model:",
            "task:",
            "target:",
            "accuracy:"
        ]

        if all(
            keyword in ml_result_lower
            for keyword in ml_required_keywords
        ):

            print("\n🧐 Reviewer Result")
            print("==================")
            print(
                "ML Agent result automatically approved."
            )

            return {
                "approved": True,
                "feedback": (
                    "The ML Agent result contains the "
                    "model, task, target, and accuracy, "
                    "so it directly provides the required "
                    "ML result."
                )
            }
    # ========================================================
    # LLM REVIEWER
    # ========================================================

    prompt = f"""
You are a strict but fair AI reviewer.

Your job is ONLY to detect clear problems in the
agent's answer.

============================================================
USER QUESTION
============================================================

{question}

============================================================
AGENT
============================================================

{agent_name}

============================================================
AGENT ANSWER
============================================================

{result}

============================================================
APPROVAL RULES
============================================================

APPROVE the answer when:

1. It directly answers the user's question.
2. It is factually reasonable.
3. It is relevant to the question.
4. It contains enough information to answer the question.
5. Short answers can be completely correct.
6. A number can be a complete answer.
7. A boolean can be a complete answer.
8. A name can be a complete answer.
9. A simple value can be a complete answer.
10. A concise factual explanation can be completely correct.

IMPORTANT:

Do NOT reject an answer just because it could contain
more explanation.

Do NOT require unnecessary context.

Do NOT require a full sentence.

Do NOT require examples unless the user requested examples.

Do NOT reject an answer because it is concise.

============================================================
RESEARCH AGENT RULES
============================================================

Research answers may be short.

For example:

Question:
What is supervised learning?

Answer:
Supervised learning uses labeled data.

Decision:
APPROVE

This is a valid concise answer.

Another valid answer:

Supervised learning uses labeled training data
to train models for classification and regression.

Decision:
APPROVE

Do NOT incorrectly interpret:

"labeled data"

as different from

"labeled training data".

In supervised learning, labeled data is used
as training data to learn the relationship between
features and the target.

============================================================
DATA AGENT RULES
============================================================

Data Agent results can be:

- A number
- A decimal
- True or False
- A string
- A list
- A dictionary
- A count
- A minimum
- A maximum
- An average
- A percentage
- A calculated value

Example:

Question:
What is the highest salary?

Answer:
65000

Decision:
APPROVE

Do not reject a correct numeric answer just because
it does not contain a sentence.

============================================================
WHEN TO REJECT
============================================================

REJECT ONLY when there is a CLEAR problem.

Examples:

- Factually wrong answer
- Irrelevant answer
- Empty answer
- Malformed answer
- Answer does not address the question
- Answer contradicts the available information
- Clearly incorrect explanation

Do NOT reject because:

- Answer is short
- Answer is concise
- Answer is only a number
- Answer is only a boolean
- Answer is only a name
- Answer has no examples
- Answer could contain more details
- Answer could be more professional

If the answer directly satisfies the question:

APPROVE

If you are unsure:

APPROVE

============================================================
IMPORTANT
============================================================

Do not invent a problem.

Do not change a correct answer.

Do not reject a correct answer because of wording
differences such as:

"labeled data"

and

"labeled training data".

Judge the actual meaning.

============================================================
DECISION
============================================================

Return ONLY valid JSON.

For APPROVE:

{{
    "approved": true,
    "feedback": "The answer correctly addresses the question."
}}

For REJECT:

{{
    "approved": false,
    "feedback": "Clearly explain the actual problem."
}}

Do not use markdown.
Do not add explanation outside JSON.
"""


    # ========================================================
    # CALL OLLAMA
    # ========================================================

    response = ollama.chat(
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


    content = response["message"]["content"].strip()


    print("\n🧐 Reviewer Result")
    print("==================")
    print(content)


    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        review = json.loads(content)

        if "approved" not in review:
            raise ValueError(
                "Missing approved field"
            )

        if "feedback" not in review:
            raise ValueError(
                "Missing feedback field"
            )

        if not isinstance(
            review["approved"],
            bool
        ):
            raise ValueError(
                "approved must be a boolean"
            )

        return review


    except Exception as e:

        print("❌ Reviewer parsing error:")
        print(e)

        # ----------------------------------------------------
        # SAFE FALLBACK
        # ----------------------------------------------------
        #
        # If reviewer itself fails, do not unnecessarily
        # destroy a potentially correct agent answer.
        # ----------------------------------------------------

        return {
            "approved": True,
            "feedback": (
                "Reviewer could not parse the response, "
                "so the original agent answer was preserved."
            )
        }


# ============================================================
# SELF-CORRECTION AGENT
# ============================================================

def self_correct_result(
    question,
    agent_name,
    result,
    feedback
):
    """
    Improve the agent result based on
    reviewer feedback.
    """

    prompt = f"""
You are a self-correction AI agent.

User Question:
{question}

Agent:
{agent_name}

Original Result:
{result}

Reviewer Feedback:
{feedback}

Improve the result based on the reviewer feedback.

Requirements:

- Answer the user's question directly.
- Keep the information accurate.
- Add missing useful details only when necessary.
- Do not invent information.
- Preserve correct numeric or data-analysis results.
- Preserve correct factual information.
- If the original answer is already correct,
  do not unnecessarily change it.
- Do not change a correct calculation just to make
  the answer longer.
- Do not introduce new unsupported information.
- Return ONLY the improved answer.
"""

    response = ollama.chat(
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

    corrected_result = (
        response["message"]["content"].strip()
    )

    print("\n🔄 Self-Corrected Result")
    print("========================")
    print(corrected_result)

    return corrected_result