**Legalassist AI** ⚖️

The challenge is the Information Barrier in the Judiciary that prevents citizens from understanding their own legal outcomes. 
Specifically: Court judgments are inaccessible to the public due to complex legal jargon and language diversity.
 
This barrier leads to: 
1. Lack of trust in the judicial system. 
2. Citizen dependency on expensive, slow intermediaries for basic case updates.

It must be solved by an automated, multilingual, plain-language translation layer applied to final judgment documents.

---

## 🚀 Recent Updates: Multi-Tenant B2B Platform
LegalAssist AI has evolved into a fully functional **B2B Multi-Tenant Platform**, supporting Law Firms and Legal Teams. 

### 🏢 Enterprise & Team Collaboration
- **Shared Workspaces**: Collaborate with your team on complex cases.
- **Role-Based Access Control (RBAC)**: Manage permissions for Partners, Associates, and Paralegals.
- **Organization Scoping**: Securely isolate data between different law firms.
- **Audit Logging**: Full compliance tracking for all sensitive actions.

---



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

## CLI Tool for Batch Processing

LegalEase AI now supports command-line processing for legal aid teams handling many judgments each day.

### Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
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
streamlit run app.py
```

---

## 🛠️ Key Modules & Features

### 🏢 Team Management
A comprehensive administration dashboard for Law Firms. Manage your organization, invite team members, and monitor compliance via audit logs.

### 🔌 REST API & SDK
Integrate LegalAssist AI into your existing workflows using our REST API. We provide a ready-to-use Python SDK located in the `sdk/` directory.

### 📅 Deadline Tracker
Never miss an appeal deadline. Our automated scheduler sends notifications via Email and SMS to the relevant team members.

### 📊 Analytics Engine
Analyze judge performance and case outcomes with our built-in analytics engine. Make data-driven decisions for your clients.

---

**Note:** The application structure has been updated to use a multi-page Streamlit layout.

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

---

## 📚 Documentation & Resources

Explore the following documents to learn more about the project:

- 🏗️ **[Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)** - Visual breakdown of the system components.
- 📖 **[Instructions](INSTRUCTIONS.md)** - Comprehensive guide to the codebase and file roles.
- 🤝 **[Contributing Guidelines](CONTRIBUTING.md)** - How to get involved and contribute to LegalAssist AI.
- ✅ **[Completion Checklist](COMPLETION_CHECKLIST.md)** - Tracking development milestones and tasks.
- 🛠️ **[Language Conversion Fix](LANGUAGE_CONVERSION_FIX.md)** - Technical details on how we handle multilingual translations.
- ⚖️ **[License](LICENSE)** - Information about the project's legal usage and distribution.

---

*“Justice delayed is justice denied. We make justice understandable.”* ⚖️
