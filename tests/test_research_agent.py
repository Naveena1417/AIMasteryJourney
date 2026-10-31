from agents.research_agent import research_question


def test_research_agent():
    result = research_question("What is supervised learning?")

    assert result is not None
    assert "answer" in result
    assert "sources" in result

    assert len(result["answer"]) > 0
    assert len(result["sources"]) > 0