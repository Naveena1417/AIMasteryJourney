
import streamlit as st
import ollama

from rag.hybrid_retriever import (
    build_hybrid_retriever,
    hybrid_search
)

from agents.data_agent import (
    load_dataset,
    generate_code,
    execute_code,
    self_correct
)

from agents.ml_agent import (
    load_ml_dataset,
    train_classification_model,
    analyze_ml_request_with_llm
)

from agents.supervisor import (
    route_question
)

from agents.reviewer import (
    review_result,
    self_correct_result
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research ML Crew",
    page_icon="🤖",
    layout="wide"
)


MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = "http://host.docker.internal:11434"


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Research ML Crew")

st.write(
    "Supervisor + Research + Data + ML Agents powered by Ollama"
)


# ============================================================
# INITIALIZE RAG
# ============================================================

@st.cache_resource
def initialize_retriever():
    return build_hybrid_retriever()


chunks, faiss_index, bm25_index = initialize_retriever()


# ============================================================
# INITIALIZE DATASET
# ============================================================

@st.cache_data
def initialize_dataset():
    return load_dataset("data/sample.csv")


dataset = initialize_dataset()


# ============================================================
# INITIALIZE ML DATASET
# ============================================================

@st.cache_data
def initialize_ml_dataset():
    return load_ml_dataset("data/ml_sample.csv")


ml_dataset = initialize_ml_dataset()


# ============================================================
# USER QUESTION
# ============================================================

st.header("💬 Ask Your Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is supervised learning?"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    try:

        # ====================================================
        # SUPERVISOR
        # ====================================================

        with st.spinner(
            "🧠 Supervisor is analyzing your question..."
        ):

            selected_agent = route_question(question)


        st.info(
            f"🧭 Supervisor selected: "
            f"**{selected_agent.upper()} Agent**"
        )


        # ====================================================
        # RESEARCH AGENT
        # ====================================================

        if selected_agent == "research":

            st.header("📚 Research Agent")


            # ------------------------------------------------
            # HYBRID RAG SEARCH
            # ------------------------------------------------

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                results = hybrid_search(
                    question,
                    chunks,
                    faiss_index,
                    bm25_index,
                    top_k=3
                )


            # ------------------------------------------------
            # BUILD CONTEXT
            # ------------------------------------------------

            context_parts = []

            for result in results:

                context_parts.append(
                    f"Source: {result['filename']}\n"
                    f"Content:\n{result['content']}"
                )


            context = "\n\n".join(context_parts)


            # ------------------------------------------------
            # LLM PROMPT
            # ------------------------------------------------

            prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer is not available in the
context, say:

"I could not find this information
in the knowledge base."

Do not make up information.

Context:
--------------------
{context}
--------------------

User Question:
{question}

Answer:
"""


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "🤖 Generating research answer..."
            ):

                response = ollama.chat(
                    model=MODEL_NAME,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )


            answer = response["message"]["content"]


            # ------------------------------------------------
            # ORIGINAL ANSWER
            # ------------------------------------------------

            st.subheader("🤖 Answer")

            st.write(answer)


            # =================================================
            # REVIEWER AGENT
            # =================================================

            st.subheader("🧐 Reviewer Agent")


            with st.spinner(
                "🧐 Reviewing the answer..."
            ):

                review = review_result(
                    question,
                    selected_agent,
                    answer
                )


            # ------------------------------------------------
            # REVIEW RESULT
            # ------------------------------------------------

            if review["approved"]:

                st.success(
                    "✅ Reviewer approved the answer."
                )

            else:

                st.warning(
                    "⚠️ Reviewer found that "
                    "the answer needs improvement."
                )

                st.write(
                    f"**Reviewer Feedback:** "
                    f"{review['feedback']}"
                )


                # =============================================
                # SELF-CORRECTION
                # =============================================

                with st.spinner(
                    "🔄 Self-correcting the answer..."
                ):

                    corrected_answer = self_correct_result(
                        question,
                        selected_agent,
                        answer,
                        review["feedback"]
                    )


                st.subheader(
                    "🔄 Corrected Answer"
                )

                st.write(corrected_answer)


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.subheader("📚 Sources")


            for rank, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**Rank {rank}** — "
                    f"📄 {result['filename']} "
                    f"| Chunk {result['chunk_id']} "
                    f"| RRF Score: "
                    f"{result['score']:.4f}"
                )


        # ====================================================
        # DATA AGENT
        # ====================================================

        elif selected_agent == "data":

            st.header("📊 Data Agent")

            if dataset is not None:

                st.subheader("📋 Dataset")
                st.dataframe(dataset)

                # ------------------------------------------------
                # GENERATE CODE
                # ------------------------------------------------

                with st.spinner(
                    "🤖 Generating Pandas code..."
                ):

                    code = generate_code(
                        question,
                        dataset.columns.tolist()
                    )

                st.subheader(
                    "🐍 Generated Pandas Code"
                )

                st.code(
                    code,
                    language="python"
                )

                # ------------------------------------------------
                # EXECUTE CODE
                # ------------------------------------------------

                with st.spinner(
                    "⚙️ Executing code..."
                ):

                    result = execute_code(
                        dataset,
                        code
                    )

                # ------------------------------------------------
                # SELF-CORRECTION FOR EXECUTION ERROR
                # ------------------------------------------------

                if (
                    isinstance(result, str)
                    and result.startswith("Error")
                ):

                    st.warning(
                        "⚠️ Execution failed. "
                        "Starting self-correction..."
                    )

                    with st.spinner(
                        "🔄 Self-correcting..."
                    ):

                        corrected_result = self_correct(
                            dataset,
                            question,
                            code,
                            result
                        )

                    if (
                        isinstance(
                            corrected_result,
                            str
                        )
                        and corrected_result.startswith("❌")
                    ):

                        st.error(
                            corrected_result
                        )

                    else:

                        st.subheader(
                            "🔄 Corrected Code"
                        )

                        st.code(
                            corrected_result,
                            language="python"
                        )

                        with st.spinner(
                            "⚙️ Executing corrected code..."
                        ):

                            final_result = execute_code(
                                dataset,
                                corrected_result
                            )

                        st.subheader(
                            "📊 Data Agent Result"
                        )

                        st.success(
                            str(final_result)
                        )

                        # -----------------------------------------
                        # REVIEW CORRECTED RESULT
                        # -----------------------------------------

                        st.subheader(
                            "🧐 Reviewer Agent"
                        )

                        with st.spinner(
                            "🧐 Reviewing the corrected result..."
                        ):

                            review = review_result(
                                question,
                                "data",
                                str(final_result)
                            )

                        if review["approved"]:

                            st.success(
                                "✅ Reviewer approved the corrected result."
                            )

                        else:

                            st.warning(
                                "⚠️ Reviewer found that "
                                "the corrected result needs improvement."
                            )

                            st.write(
                                f"**Reviewer Feedback:** "
                                f"{review['feedback']}"
                            )

                            with st.spinner(
                                "🔄 Self-correcting the final result..."
                            ):

                                corrected_final_result = (
                                    self_correct_result(
                                        question,
                                        "data",
                                        str(final_result),
                                        review["feedback"]
                                    )
                                )

                            st.subheader(
                                "🔄 Final Corrected Result"
                            )

                            st.success(
                                corrected_final_result
                            )

                # ------------------------------------------------
                # SUCCESSFUL EXECUTION
                # ------------------------------------------------

                else:

                    st.subheader(
                        "📊 Data Agent Result"
                    )

                    st.success(
                        str(result)
                    )

                    # ---------------------------------------------
                    # REVIEWER AGENT
                    # ---------------------------------------------

                    st.subheader(
                        "🧐 Reviewer Agent"
                    )

                    with st.spinner(
                        "🧐 Reviewing the Data Agent result..."
                    ):

                        review = review_result(
                            question,
                            "data",
                            str(result)
                        )

                    # ---------------------------------------------
                    # REVIEW RESULT
                    # ---------------------------------------------

                    if review["approved"]:

                        st.success(
                            "✅ Reviewer approved the Data Agent result."
                        )

                    else:

                        st.warning(
                            "⚠️ Reviewer found that "
                            "the Data Agent result needs improvement."
                        )

                        st.write(
                            f"**Reviewer Feedback:** "
                            f"{review['feedback']}"
                        )

                        # -----------------------------------------
                        # SELF-CORRECTION
                        # -----------------------------------------

                        with st.spinner(
                            "🔄 Self-correcting the result..."
                        ):

                            corrected_result = (
                                self_correct_result(
                                    question,
                                    "data",
                                    str(result),
                                    review["feedback"]
                                )
                            )

                        st.subheader(
                            "🔄 Corrected Result"
                        )

                        st.success(
                            corrected_result
                        )

            else:

                st.error(
                    "❌ Dataset could not be loaded."
                )


        # ====================================================
        # ML AGENT
        # ====================================================

        elif selected_agent == "ml":

            st.header("🤖 ML Agent")

            if ml_dataset is not None:

                st.subheader("📋 ML Dataset")

                st.dataframe(
                    ml_dataset
                )


                # ------------------------------------------------
                # LLM ANALYSIS
                # ------------------------------------------------

                with st.spinner(
                    "🧠 ML Agent is analyzing the request..."
                ):

                    analysis = (
                        analyze_ml_request_with_llm(
                            question,
                            ml_dataset.columns.tolist()
                        )
                    )


                task = analysis["task"]

                target_column = (
                    analysis["target_column"]
                )

                selected_model = (
                    analysis["model"]
                )


                # ------------------------------------------------
                # ML ANALYSIS DISPLAY
                # ------------------------------------------------

                st.subheader(
                    "🧠 ML Analysis"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Task",
                        task
                    )


                with col2:

                    st.metric(
                        "Target",
                        target_column
                    )


                with col3:

                    st.metric(
                        "Model",
                        selected_model
                    )


                # ------------------------------------------------
                # TRAIN MODEL
                # ------------------------------------------------

                if (
                    task == "classification"
                    and target_column is not None
                ):

                    with st.spinner(
                        f"🤖 Training {selected_model}..."
                    ):

                        model, accuracy = (
                            train_classification_model(
                                ml_dataset,
                                target_column,
                                model_name=selected_model
                            )
                        )


                    st.subheader(
                        "🏆 ML Agent Result"
                    )


                    result_col1, result_col2 = (
                        st.columns(2)
                    )


                    with result_col1:

                        st.metric(
                            "Selected Model",
                            selected_model
                        )


                    with result_col2:

                        st.metric(
                            "Accuracy",
                            f"{accuracy:.2f}"
                        )


                    st.success(
                        "✅ Model training and "
                        "evaluation completed!"
                    )


                    # =================================================
                    # REVIEWER AGENT
                    # =================================================

                    st.subheader(
                        "🧐 Reviewer Agent"
                    )


                    ml_result = (
                        f"Model: {selected_model} | "
                        f"Task: {task} | "
                        f"Target: {target_column} | "
                        f"Accuracy: {accuracy:.2f}"
                    )


                    with st.spinner(
                        "🧐 Reviewing the ML Agent result..."
                    ):

                        review = review_result(
                            question,
                            "ml",
                            ml_result
                        )


                    # ------------------------------------------------
                    # REVIEW RESULT
                    # ------------------------------------------------

                    if review["approved"]:

                        st.success(
                            "✅ Reviewer approved the ML Agent result."
                        )

                    else:

                        st.warning(
                            "⚠️ Reviewer found that "
                            "the ML Agent result needs improvement."
                        )

                        st.write(
                            f"**Reviewer Feedback:** "
                            f"{review['feedback']}"
                        )


                        # ---------------------------------------------
                        # SELF-CORRECTION
                        # ---------------------------------------------

                        with st.spinner(
                            "🔄 Self-correcting the ML result..."
                        ):

                            corrected_result = (
                                self_correct_result(
                                    question,
                                    "ml",
                                    ml_result,
                                    review["feedback"]
                                )
                            )


                        st.subheader(
                            "🔄 Corrected ML Result"
                        )

                        st.success(
                            corrected_result
                        )


                else:

                    st.error(
                        "❌ Unsupported ML task "
                        "or invalid target column."
                    )


            else:

                st.error(
                    "❌ ML dataset could not be loaded."
                )


        # ====================================================
        # UNKNOWN AGENT
        # ====================================================

        else:

            st.error(
                f"❌ Unknown agent selected: "
                f"{selected_agent}"
            )


    # ========================================================
    # GLOBAL ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Application error"
        )

        st.error(
            str(e)
        )
