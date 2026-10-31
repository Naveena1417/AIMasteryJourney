from agents.reviewer import review_result


def test_reviewer_research_approved():
    result = review_result(
        "What is supervised learning?",
        "research",
        "Supervised learning uses labeled data."
    )

    assert result["approved"] is True


def test_reviewer_data_approved():
    result = review_result(
        "What is the highest salary?",
        "data",
        "The highest salary is 65000"
    )

    assert result["approved"] is True


def test_reviewer_data_rejected():
    result = review_result(
        "What is the highest salary?",
        "data",
        "The highest salary is 99999"
    )

    assert result["approved"] is False