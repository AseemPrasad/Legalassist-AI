## Live Demo

**Try it out:** [View Project on GitHub](https://github.com/AseemPrasad/Legalassist-AI) *(Live deployment coming soon)*


> The live deployment URL will be updated once the production hosting is configured. In the meantime, please follow the [installation instructions](#developer-setup--running-tests) below to run the project locally.



**Legalassist AI (Unified Legal Information Platform)**

The challenge is the Information Barrier in the Judiciary that prevents citizens from understanding their own legal outcomes. 
Specifically: Court judgments are inaccessible to the public due to complex legal jargon and language diversity.
 
This barrier leads to: 
1. Lack of trust in the judicial system. 
2. Citizen dependency on expensive, slow intermediaries for basic case 
updates.

 It must be solved by an automated, multilingual, plain-language 
translation layer applied to final judgment documents.


Legalassist AI
 An AI-powered, multilingual translation engine that converts complex, jargon-filled judicial judgments into three key points of clear, actionable information for the citizen.
 Addresses the Problem: It directly dismantles the Information Barrier (our defined problem) by instantly providing clarity and eliminating the reliance on expensive, slow intermediaries for basic understanding.
 This solution directly breaks the language and jargon barrier by providing instant clarity and removing dependence on expensive intermediaries for basic understanding.

  The entire process is designed to be completed in less than 60 seconds. The interface requires only one significant action from the user (upload/paste), and the system handles the entire complex process of legal interpretation and translation, demonstrating true simplification



**Impact on the Target Audience (The Citizen Litigant)**

 The core impact is shifting the citizen's status from a dependent bystander to an informed participant.Before Citizens wait years for closure and cannot navigate courts due to language and cost barriers, relying solely on 
intermediaries for basic updates. The judiciary is stuck with manual records and PDFs, leaving the citizen confused.

 After The solution eliminates the information gap, leading to:
 
 Emotional Relief & Clarity: The primary source of post-judgment anxiety (not knowing what the document means) is removed by providing instant, actionable clarity.
 
 Zero Dependency Cost: Citizens are no longer forced to pay or wait for legal aid/middlemen merely to understand the outcome of their case, directly addressing the cost barrier.
 
 Trust Building: By offering tamper-proof clarity, the solution begins to rebuild trust in the legal system, countering the perceived absence of transparency.
 
 The benefits are defined by the direct, automated replacement of flawed manual processes.
 
 Automation of Clarity (AI Advantage): The system auto-generates plain-language judgment explainers instantly. This is a quantum leap over the slow, manual process of a lawyer explaining a complex document.
 
 Accessibility (Digital Divide Bridge): By instantly converting legal jargon into local language summaries, the solution bridges the Digital Divide and promotes inclusive justice for ordinary people who cannot navigate the courts due to language



## Live Demo

**Try it out:** [LegalAssist AI Live App](https://legalassist-ai.example.com)

> The live deployment URL will be updated once the production hosting is configured. In the meantime, please follow the [installation instructions](#developer-setup--running-tests) below to run the project locally.
## Developer Setup & Running Tests

### Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or source venv/bin/activate on Linux/macOS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-notifications.txt
   ```
3. Run environment validation:
   ```bash
   python scripts/validate_env.py
   ```

### Running Tests
To run the test suite:
```bash
pytest
```

## CLI Tool for Batch Processing

LegalEase AI now supports command-line processing for legal aid teams handling many judgments each day.

### Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
# Optional: Install Twilio and SendGrid for notifications and OTP delivery
pip install -r requirements-notifications.txt
```

3. Set API environment variables:

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_key_here"
$env:OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### CLI Commands

Show full help:

```bash
python cli.py --help
```

Process a single file:

```bash
python cli.py process --file judgment.pdf --language Hindi
```

Process a scanned/image PDF using OCR (Hindi + English):

```bash
python cli.py process --file scanned_judgment.pdf --enable-ocr --ocr-languages eng+hin
```

Batch process a folder (parallel workers):

```bash
python cli.py batch --folder ./documents --output results.csv --workers 4
```

Alias form (also supported):

```bash
python cli.py process_batch --input ./judgments_folder --output ./results.csv
```

### Key Features

- Reads all PDFs from a folder
- Generates summary and remedies advice for each PDF
- Parallel processing (`--workers`, default `4`)
- Resume capability via checkpoint file
- Per-file error handling (one failure does not stop the run)
- Real-time progress bar with status and running cost
- Exports to CSV/JSON (`--format csv|json|both`, default `both`)
- Language controls: fixed (`--language Hindi`) or auto-detect (`--language auto`)
- OCR fallback for scanned PDFs (`--enable-ocr`)
- OCR language packs for local scripts (`--ocr-languages eng+hin`)
- OCR quality signal via extraction confidence in output

### Resume Behavior

- Default mode resumes automatically.
- Checkpoint path defaults to `<output>.checkpoint.jsonl`.
- Successful files in checkpoint are skipped on re-run.
- Use `--no-resume` to start from scratch.

### Output Format

The exported CSV/JSON includes one record per PDF with:

- `file_name`, `file_path`
- `status` (`success` or `error`), `error`
- `language`
- `summary`
- `what_happened`, `can_appeal`, `appeal_days`, `appeal_court`, `cost_estimate`, `first_action`, `deadline`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `api_cost_usd` (estimated)
- `duration_seconds`, `processed_at`

### Cost Estimation

CLI prints total tokens and total estimated API cost at the end of batch runs.

By default, cost per token is `0.0` unless configured. Set these flags to match your provider pricing:

```bash
python cli.py batch \
  --folder ./documents \
  --output ./results.csv \
  --workers 4 \
  --prompt-cost-per-1k 0.0002 \
  --completion-cost-per-1k 0.0002
```

Estimated cost formula:

$$
  \\text{total_cost_usd} = \\left(\\frac{\\text{prompt_tokens}}{1000}\\right)\\cdot p + \\left(\\frac{\\text{completion_tokens}}{1000}\\right)\\cdot c
$$

where $p$ and $c$ are prompt/completion USD rates per 1K tokens.

### Example: 10+ PDFs

```bash
python cli.py batch --folder ./tests/samples --output ./outputs/results.csv --workers 4 --recursive
```

This command is suitable for validating a 10+ file run with concurrency, checkpoint resume, and export outputs.

## Continuous Integration

The repository CI runs a Python version matrix on GitHub Actions:

- Python `3.10`, `3.11`, and `3.12`
- `pip` caching via `actions/setup-python`
- `.pytest_cache` reuse between runs
- `pytest-xdist` parallel execution
- `pytest-rerunfailures` retry-based flaky-test detection

If a test flakes and passes on retry, the workflow uploads a `ci-artifacts-*.zip` artifact containing the pytest log and the flaky node IDs.

## 🔍 Metadata Extraction

LegalEase AI can automatically extract key case metadata from judgment documents, including:

- **Case Number/Identifier** - Court case number or citation
- **Filing Date** - When the case was filed
- **Judgment Date** - When judgment was delivered
- **Presiding Judge/Bench** - Judge(s) who delivered the judgment
- **Court Name** - Which court heard the case
- **Party Names** - Petitioner, respondent, plaintiff, defendant, etc.

### Quick Start

#### Single Document

```bash
python cli.py extract-metadata --file judgment.pdf
```

Output:
```json
{
  "file": "judgment.pdf",
  "metadata": {
    "case_number": "2023/456/789",
    "filing_date": "2023-01-15",
    "judgment_date": "2023-05-20",
    "judge_name": "Justice Rajesh Sharma",
    "court_name": "Delhi High Court",
    "petitioner": "Ram Kumar Singh",
    "respondent": "Government of India",
    "parties": ["Ram Kumar Singh", "Government of India"],
    "confidence_score": 0.85,
    "extraction_method": "pattern"
  }
}
```

#### Save to File

```bash
python cli.py extract-metadata --file judgment.pdf --output metadata.json
```

#### With LLM Enhancement

For complex or unstructured documents, enable LLM-based extraction:

```bash
python cli.py extract-metadata --file judgment.pdf --enable-llm
```

#### Scanned Documents (with OCR)

```bash
python cli.py extract-metadata --file scanned_judgment.pdf --enable-ocr --ocr-languages eng+hin
```

### Python API Usage

```python
from document_parser import extract_metadata

# Basic extraction
metadata = extract_metadata('judgment.pdf')
print(f"Case: {metadata.case_number}")
print(f"Judge: {metadata.judge_name}")
print(f"Confidence: {metadata.confidence_score}")

# With LLM enhancement
metadata = extract_metadata('judgment.pdf', enable_llm=True)

# Convert to JSON
print(metadata.to_json())

# Convert to dictionary
data = metadata.to_dict()
```

### Batch Processing

```python
from pathlib import Path
from document_parser import extract_metadata_batch

# Extract from multiple files
doc_paths = list(Path('judgments/').glob('*.pdf'))
results = extract_metadata_batch(doc_paths, enable_llm=False)

# Access results
for result in results:
    print(f"File: {result['document_path']}")
    print(f"Case: {result['case_number']}")
    print(f"Confidence: {result['confidence_score']}")
```

### Supported Formats

- **Date Formats**: DD-MM-YYYY, DD/MM/YYYY, DD MMM YYYY, YYYY-MM-DD
- **Case Numbers**: Various Indian court formats (e.g., 2023/123, WP-1234-2023, CA NO. 456 OF 2023)
- **Courts**: Supreme Court, High Courts, District Courts, Sessions Courts, Family Courts, etc.
- **Languages**: English, Hindi, and other Indian languages (with OCR enabled)

### Confidence Scoring

The confidence score (0.0-1.0) indicates how complete the extraction was:
- **0.8-1.0**: All major fields extracted successfully
- **0.5-0.8**: Most fields extracted, some missing
- **0.2-0.5**: Partial extraction, many fields missing
- **< 0.2**: Minimal extraction, document may be incompatible

### API Reference

```python
extract_metadata(
    document_input,           # Path, text string, or file-like object
    enable_llm=False,         # Use LLM for complex extraction
    enable_ocr=False,         # Enable OCR for scanned documents
) -> CaseMetadata
```

Returns a `CaseMetadata` object with:
- `case_number`: Extracted case identifier
- `filing_date`: YYYY-MM-DD format
- `judgment_date`: YYYY-MM-DD format
- `judge_name`: Presiding judge/bench
- `court_name`: Court name
- `petitioner`: First party name
- `respondent`: Second party name
- `parties`: List of all party names
- `confidence_score`: Extraction confidence (0.0-1.0)
- `extraction_method`: "pattern" or "llm"

### Troubleshooting

**No metadata extracted:**
- Ensure the document contains readable text
- Try enabling OCR for scanned documents: `--enable-ocr`

**Date format issues:**
- Supported formats: DD-MM-YYYY, DD/MM/YYYY, DD MMM YYYY
- Other formats may need LLM extraction: `--enable-llm`

**Poor confidence score:**
- Document may have non-standard formatting
- Enable LLM extraction for better results: `--enable-llm`
- Provide complete judgment documents (metadata is typically at the start)

### Testing

Run metadata extraction tests:

```bash
pytest tests/test_document_parser.py -v
```

Test with sample judgments:

```bash
python cli.py extract-metadata --file tests/samples/criminal/guilty/sample_001.pdf --output debug.json
```
## 📊 Analytics Dashboard

LegalEase AI now includes a comprehensive analytics dashboard that tracks case outcomes and helps users make informed appeal decisions.

### Features

📈 **Case Analytics**
- Track all processed cases (anonymized)
- Monitor success rates by jurisdiction, court, and judge
- Identify trends and patterns in case outcomes

🎯 **Appeal Success Estimator**
- Estimate your appeal success probability based on similar cases
- Get cost and time estimates
- See confidence levels based on data quantity

📝 **Outcome Feedback Form**
- Report your case results and appeal outcomes
- Help improve predictions for future users
- Anonymous and confidential

📊 **Judge Performance Analytics**
- See which judges have higher appeal success rates
- Regional comparisons
- Identify high-performing courts

### Getting Started

#### 1. Initialize Analytics Database
```bash
python -c "from database import init_db; init_db()"
```

#### 2. Generate Sample Data (Optional, for testing)
```bash
# Generate 100 sample cases
python scripts/generate_sample_analytics_data.py 100

# Generate more cases for better estimates
python scripts/generate_sample_analytics_data.py 500

# Clear sample data when done
python scripts/generate_sample_analytics_data.py clear
```

#### 3. Start the App
```bash
streamlit run pages/0_Home.py
```

**Note:** The main application entry point is `pages/0_Home.py`. 
- Multi-page structure with Streamlit's automatic routing
- Pages located in `pages/` directory:
  - `0_Home.py` - Judgment analysis (main feature)
  - `1_Deadlines.py` - Appeal deadline management
  - `2_History.py` - Notification history
  - `3_Settings.py` - User preferences
- Core utilities extracted to `core/app_utils.py`
- Legacy files (`app.py`, `app_integrated.py`) have been deprecated and consolidated into this unified structure

#### 4. Access the Pages

After uploading a judgment:
- **Analytics Dashboard** → View case statistics and trends
- **Appeal Estimator** → Get your appeal success probability
- **Report Outcome** → Submit feedback about your case

### How Appeal Success Estimation Works

1. **Enter your case details** (type, jurisdiction, court, judge)
2. **System finds similar cases** from the database
3. **Calculates success rate** based on similar cases
4. **Adjusts for your specifics** (decision clarity, case value, etc.)
5. **Returns probability** with confidence level

**Example:**
```
Case: Civil case in Delhi High Court before Justice Sharma

Similar Cases Found: 23
Appeal Success Rate: 22%
Confidence: Medium

Estimated Cost: ₹12,000 - ₹25,000
Typical Duration: 12-24 months
```

### Privacy & Anonymization

✅ **What's protected:**
- No case numbers or party names stored
- No identifiable personal information
- User feedback is anonymous
- Data aggregated before display

✅ **What's tracked (anonymized):**
- Case type, jurisdiction, court, judge
- Outcomes (won/lost/settlement)
- Appeal filing and success rates
- Timeline data

### Data Available

The analytics dashboard works best with real case data. Sample data is provided for testing:
- 100+ sample cases across 10 jurisdictions
- Realistic success rates and timelines
- Multiple case types (Civil, Criminal, Family, Commercial, Labor)

### Analytics Engine

The system uses:
- **Similarity Matching**: Finds cases similar to yours (50+ parameters)
- **Statistical Analysis**: Calculates success rates by demographics
- **Confidence Scoring**: Rates estimate reliability based on data quantity
- **Trend Analysis**: Identifies regional and judge-specific patterns

### For Developers

See [ANALYTICS.md](ANALYTICS.md) for:
- Detailed architecture
- API reference
- Database schema
- Sample data generation
- Integration examples
