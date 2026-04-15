**Legalassist AI**

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


