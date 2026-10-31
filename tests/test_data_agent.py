from agents.data_agent import load_dataset, answer_question


def test_load_dataset():
    data = load_dataset("data/sample.csv")

    assert data is not None
    assert len(data) == 5


def test_highest_salary():
    data = load_dataset("data/sample.csv")

    result = answer_question(
        data,
        "What is the highest salary?"
    )

    assert result == "The highest salary is 65000"


def test_average_salary():
    data = load_dataset("data/sample.csv")

    result = answer_question(
        data,
        "What is the average salary?"
    )

    assert result == "The average salary is 48000.00"