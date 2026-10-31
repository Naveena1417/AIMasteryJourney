from agents.supervisor import route_question
from agents.data_agent import load_dataset, answer_question
from agents.reviewer import review_result
from agents.research_agent import research_question
from agents.ml_agent import analyze_ml_request_with_llm, train_classification_model

# ============================================================
# SELF-CORRECTION
# ============================================================

def self_correct_data_result(data, question, feedback):

    print("\n🔄 Self-Correction")
    print("==================")

    print("Reviewer Feedback:")
    print(feedback)

    corrected_result = answer_question(
        data,
        question
    )

    return corrected_result

# ============================================================
# CREW ORCHESTRATOR
# ============================================================

def run_crew(question):

    print("\n" + "=" * 60)
    print("🤖 AI RESEARCH ML CREW")
    print("=" * 60)

    print("\n👤 User Question")
    print("================")
    print(question)

    # --------------------------------------------------------
    # STEP 1 — Supervisor
    # --------------------------------------------------------

    agent_name = route_question(question)

    print("\n🧭 Supervisor Decision")
    print("=====================")
    print(f"Selected Agent: {agent_name}")

    # --------------------------------------------------------
    # STEP 2 — Data Agent
    # --------------------------------------------------------

    if agent_name == "data":

        data = load_dataset("data/sample.csv")

        if data is None:
            return "❌ Dataset could not be loaded."

        result = answer_question(data, question)

        print("\n📊 Data Agent Result")
        print("====================")
        print(result)

        # --------------------------------------------------------
        # STEP 3 — Reviewer Agent
        # --------------------------------------------------------

        review = review_result(
            question,
            agent_name,
            result
        )

        print("\n🧐 Reviewer")
        print("===========")
        print(f"Approved: {review['approved']}")
        print(f"Feedback: {review['feedback']}")

        if review["approved"]:

            print("\n✅ Reviewer approved the result.")

            return result

        else:

            print("\n❌ Reviewer rejected the result.")

            corrected_result = self_correct_data_result(
                data,
                question,
                review["feedback"]
            )

            print("\n📊 Corrected Data Agent Result")
            print("=============================")
            print(corrected_result)

            return corrected_result
    # --------------------------------------------------------
    # Other agents — not connected yet
    # --------------------------------------------------------

    elif agent_name == "research":

        result = research_question(question)

        print("\n🔬 Research Agent Result")
        print("========================")
        print(result["answer"])

        print("\n📚 Sources")
        print("==========")

        for source in result["sources"]:
            print(
                f"- {source['filename']} "
                f"(score: {source['score']:.4f})"
            )
        review = review_result(
            question,
            agent_name,
            result["answer"]
        )

        print("\n🧐 Research Reviewer")
        print("====================")
        print(f"Approved: {review['approved']}")
        print(f"Feedback: {review['feedback']}")

        return result["answer"]
    elif agent_name == "ml":

        data = load_dataset("data/ml_sample.csv")

        if data is None:
            return "❌ ML dataset could not be loaded."

        analysis = analyze_ml_request_with_llm(
            question,
            data.columns.tolist()
        )

        print("\n🤖 ML Agent Analysis")
        print("====================")
        print(analysis)

        if analysis["target_column"] is None:
            return "❌ ML Agent could not identify the target column."

        result = train_classification_model(
            data,
            analysis["target_column"],
            analysis["model"]
        )

        print("\n📈 ML Agent Result")
        print("==================")
        print(result)

        review = review_result(
            question,
            agent_name,
            result
        )

        print("\n🧐 ML Reviewer")
        print("==============")
        print(f"Approved: {review['approved']}")
        print(f"Feedback: {review['feedback']}")

        return str(result)
    return "❌ Unknown agent."


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question =  "Predict whether a customer will purchase"

    result = run_crew(question)

    print("\n✅ Crew Final Result")
    print("===================")
    print(result)
