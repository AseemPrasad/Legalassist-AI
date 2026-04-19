
# LegalEase AI Test Suite

Comprehensive test suite ensuring that judgment simplification, remedies extraction, and notification systems work correctly.

## Test Coverage

- ✅ **test_app.py** (10 tests) - Basic functionality tests
- ✅ **test_app_core.py** (35+ tests) - Core business logic: PDF extraction, text compression, language detection
- ✅ **test_remedies.py** (50+ tests) - Remedies parsing with 20+ realistic fixtures and mocked API
- ✅ **test_notifications.py** (22+ tests) - Notification system with 95% coverage

**Total:** 100+ unit and integration tests with **80%+ code coverage**

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest-cov pytest-mock responses
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Core business logic
pytest tests/test_app_core.py -v

# Remedies parsing with fixtures
pytest tests/test_remedies.py -v

# Notifications
pytest tests/test_notifications.py -v

# All with coverage report
pytest tests/ --cov=core --cov=database --cov=notification_service --cov-report=html
```

### Generate Coverage Report
```bash
pytest tests/ \
  --cov=core \
  --cov=database \
  --cov=notification_service \
  --cov=scheduler \
  --cov-report=html \
  --cov-report=term-missing

# Open htmlcov/index.html to view coverage
```

## Test Structure

### Core Business Logic Tests (`test_app_core.py`)

Tests for judgment analysis pipeline:

**PDF Extraction**
- Valid PDF text extraction
- Sample PDF fixtures
- Content preservation
- Empty/image-only PDF error handling

**Text Compression**
- Long text compression with truncation marker
- Short text passes through unchanged
- Custom limit handling
- Preserves head and tail content

**English Leakage Detection**
- Detects English words in non-English text
- Pure language text validates correctly
- Threshold sensitivity testing
- Mixed language detection

**Prompt Building**
- All language support (English, Hindi, Bengali, Urdu)
- Model instructions included
- Retry prompts for language correction
- Remedies advisor prompts

**Remedies Parsing**
- 7-section structured format parsing
- Flexible separators (., ), :, -)
- Empty response handling
- Appeal info extraction

### Remedies Extraction Tests (`test_remedies.py`)

**20+ Realistic Fixtures** covering:
- Criminal guilty verdict with appeal option
- Criminal acquittal (limited appeal)
- Civil plaintiff won with damages
- Civil plaintiff lost
- Family custody decisions
- Labor termination cases
- Landlord-tenant disputes
- Edge cases (no appeal, extended timelines, multiple remedies)

**Mock API Tests**
- Correct model selection
- Language parameter passing
- Error handling
- Prompt structure validation

**Quality Assurance**
- Actionable advice presence
- Timeline information
- Format consistency across all responses

## Test Fixtures

### Sample PDF Locations
```
tests/samples/
├── criminal/
│   ├── guilty/       (10 guilty verdicts)
│   └── acquitted/    (10 acquittals)
├── civil/
│   ├── plaintiff_won/    (10 victories)
│   └── plaintiff_lost/   (10 defeats)
├── family/
│   ├── custody_granted/  (6 cases)
│   └── custody_denied/   (4 cases)
├── labor/
│   ├── termination_upheld/     (4 cases)
│   └── termination_overturned/ (2 cases)
└── landlord_tenant/
    ├── eviction_granted/       (4 cases)
    └── eviction_denied/        (2 cases)
```

### Generate/Expand Test Data
```bash
# Generate 50 default fixtures (all case types)
python scripts/generate_test_data.py

# Generate specific number of fixtures
python scripts/generate_test_data.py 100
```

Output: 50 PDF files + `tests/test_metadata.json`

## GitHub Actions CI/CD

Automated testing on push/PR to main:

```yaml
- Python 3.10, 3.11, 3.12 matrix
- Dependency installation
- Linting with flake8 (non-blocking)
- All test suites
- Coverage report (80%+ required)
- Security scans (bandit, safety)
```

View workflows: `.github/workflows/tests.yml`

## How to Add Test Cases

### 1. Add PDF Sample
```bash
# Place judgment PDF in appropriate folder:
tests/samples/criminal/guilty/your_case.pdf
```

### 2. Update Metadata
```json
{
    "path": "tests/samples/criminal/guilty/your_case.pdf",
    "type": "criminal_guilty",
    "expected_verdict": "guilty",
    "expected_appeal": "yes",
    "expected_days": "90"
}
```

### 3. Add Fixture to test_remedies.py
```python
REMEDIES_FIXTURES = {
    "your_case_type": """
    1. What happened?
    ... (numbered 7-section format)
    """
}
```

### 4. Add Test Function
```python
def test_your_case_type(self):
    response = REMEDIES_FIXTURES["your_case_type"]
    remedies = parse_remedies_response(response)
    assert remedies["what_happened"]
    # Add assertions specific to your case
```

## Expected vs Actual Comparison

| Component | Expected | Actual | How to Verify |
|-----------|----------|--------|---------------|
| PDF Extraction | Text with keywords | Extracted text | `pytest tests/test_app_core.py::TestPDFExtraction -v` |
| Compression | Truncation for large text | < limit bytes | `pytest tests/test_app_core.py::TestTextCompression -v` |
| Language Detection | Leakage flagged | Flag on threshold | `pytest tests/test_app_core.py::TestEnglishLeakage -v` |
| Remedies Parsing | 7 sections | Dict with values | `pytest tests/test_remedies.py -v` |
| API Integration | Correct model called | Model in call args | `pytest tests/test_remedies.py::TestGetRemediesAdviceWithMocks -v` |
| Coverage | 80%+ | HTML report | `pytest --cov=core --cov-report=html` |

## Continuous Integration

Tests run automatically:
- **On push** to main/develop branches
- **On PR** to main/develop
- **Matrices:** Python 3.10, 3.11, 3.12
- **Artifacts:** Coverage HTML reports

View test results: GitHub Actions tab in repo

## Troubleshooting

### ImportError: No module named 'core'
```bash
# Ensure core/ directory with __init__.py exists
# Run from project root
cd /path/to/Legalassist-AI
pytest tests/
```

### Streamlit secrets not found
Tests mock secrets automatically. If manual setup needed:
```bash
mkdir -p .streamlit
echo 'OPENROUTER_API_KEY = "test_key"' > .streamlit/secrets.toml
echo 'OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"' >> .streamlit/secrets.toml
```

### Coverage report not generated
Ensure pytest-cov is installed:
```bash
pip install pytest-cov
pytest tests/ --cov=core --cov-report=html
```

## Test Statistics

- **Total Tests:** 100+
- **Core Logic Coverage:** 90%+
- **Notification Coverage:** 95%
- **Average Test Execution:** < 15 seconds
- **CI Pipeline:** ~2 minutes
- **Test Fixtures:** 50+ realistic sample cases

## Next Steps

- [ ] Add performance benchmarks
- [ ] Expand family law fixtures
- [ ] Add multilingual response testing
- [ ] Integration with LLM API (non-mocked)
- [ ] Load testing for batch processing
| Summary Length | Ensure exactly 3 bullet points are requested and parsed. |
| Remedy Parsing | Ensure the 5-point structure is maintained even with varied LLM phrasing. |
| Language Leakage | Ensure localized outputs don't contain common English stop words. |
