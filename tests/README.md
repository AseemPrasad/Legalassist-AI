
# LegalEase AI Test Suite

This test suite ensures that the legal judgment simplification and remedies extraction logic works correctly.

## Structure

- `tests/samples/`: Contains 20+ sample judgment PDFs categorized by outcome.
- `tests/test_metadata.json`: Metadata for each test case, including paths and expected outcomes.
- `test_app.py`: The main test file using `pytest`.
- `scripts/generate_test_data.py`: Script to generate synthetic test PDFs.

## How to Run Tests

1. Install dependencies:
   ```bash
   pip install pytest fpdf2 pypdf openai streamlit
   ```

2. Run tests:
   ```bash
   pytest test_app.py
   ```

## How to Add New Test Cases

1. **Add PDF**: Place a new judgment PDF in the appropriate folder under `tests/samples/`.
2. **Update Metadata**: Add an entry to `tests/test_metadata.json`:
   ```json
   {
       "path": "tests/samples/your_case.pdf",
       "type": "criminal_guilty",
       "expected_verdict": "guilty",
       "expected_appeal": "yes",
       "expected_days": "30"
   }
   ```
3. **Run Tests**: The `test_pdf_extraction` will automatically pick up the new file.

## How to Validate Test Results

- **Automated**: The tests check for PDF extraction success, prompt string correctness, and parsing logic.
- **Manual**: Since LLM outputs can vary, use the `scripts/generate_test_data.py` to see how the app behaves with specific edge cases.
- **Mocking**: LLM calls are mocked in `test_app.py` to ensure unit tests are fast and deterministic.

## Expected vs Actual Comparison Guide

| Test Component | What to look for |
|----------------|-----------------|
| PDF Extraction | Ensure text is not empty and contains key terms. |
| Summary Length | Ensure exactly 3 bullet points are requested and parsed. |
| Remedy Parsing | Ensure the 5-point structure is maintained even with varied LLM phrasing. |
| Language Leakage | Ensure localized outputs don't contain common English stop words. |
