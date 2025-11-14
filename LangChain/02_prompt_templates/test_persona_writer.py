import pytest
from persona_writer import generate_paragraph

class TestPersonaWriter:

    def test_missing_variables(self):
        with pytest.raises(TypeError):
            generate_paragraph(tone="formal")

    def test_empty_string(self):
        try:
            generate_paragraph(tone="formal", topic="")
            generate_paragraph(tone="", topic="")
        except ValueError:
            pytest.fail("generate_paragraph raised ValueError unexpectedly")