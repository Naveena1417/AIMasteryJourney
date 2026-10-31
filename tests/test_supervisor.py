from agents.supervisor import route_question


def test_supervisor_research():
    result = route_question("What is supervised learning?")
    assert result == "research"


def test_supervisor_data():
    result = route_question("What is the highest salary?")
    assert result == "data"


def test_supervisor_ml():
    result = route_question("Predict whether a customer will purchase")
    assert result == "ml"