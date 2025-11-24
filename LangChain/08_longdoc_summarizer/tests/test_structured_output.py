import json
from langchain_core.runnables import RunnableLambda

from src.config import LongDocSummarizerSettings
from src.structured_output import SummaryQA, generate_qa_response, validate_output

class DummyChatOpenAI(RunnableLambda):
    """Runnable stub that always returns a fixed JSON payload."""
    def __init__(self, *_, **__):
        response_json = json.dumps(
            {
                "answer": "42",
                "bullets": ["item 1", "item 2", "item 3"],
                "risk_notes": ["low confidence"],
            }
        )

        def _call(_prompt: str, _config=None):
            return response_json

        super().__init__(_call)

class TestGenerateQAResponse:

    def test_generate_qa_response_returns_summaryqa(self, monkeypatch):
        """Test that generate_qa_response returns a SummaryQA object."""
        monkeypatch.setattr("src.structured_output.ChatOpenAI", DummyChatOpenAI)

        settings = LongDocSummarizerSettings(api_key="test-key")
        result = generate_qa_response("What is the answer?", "Sample summary.", settings)

        assert isinstance(result, SummaryQA)
        assert result.answer == "42"


    def test_generate_qa_response_populates_fields(self,monkeypatch):
        """Test that generate_qa_response populates the fields of the SummaryQA object."""
        monkeypatch.setattr("src.structured_output.ChatOpenAI", DummyChatOpenAI)

        settings = LongDocSummarizerSettings(api_key="test-key")
        result = generate_qa_response("Question?", "Summary text.", settings)

        assert result.bullets == ["item 1", "item 2", "item 3"]
        assert result.risk_notes == ["low confidence"]

class TestValidateOutput:

    def test_validate_output_passes_for_valid_response(self):
        """Test that validate_output passes for a valid response."""
        response = SummaryQA(
            answer="Hello",
            bullets=["a", "b", "c"],
            risk_notes=["low risk"],
        )

        is_valid, error = validate_output(response)

        assert is_valid is True
        assert error is None


    def test_validate_output_requires_answer(self):
        """Test that validate_output requires an answer."""
        response = SummaryQA(
            answer="",
            bullets=["a", "b", "c"],
            risk_notes=[],
        )

        is_valid, error = validate_output(response)

        assert is_valid is False
        assert error == "Answer is required."