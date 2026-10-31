from crew import run_crew


def test_crew_data_workflow():
    result = run_crew("What is the highest salary?")
    assert result is not None


def test_crew_research_workflow():
    result = run_crew("What is supervised learning?")
    assert result is not None


def test_crew_ml_workflow():
    result = run_crew("Predict whether a customer will purchase")
    assert result is not None