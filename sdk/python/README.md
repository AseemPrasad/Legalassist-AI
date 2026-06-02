# Python SDK for Legalassist-AI

Complete async Python SDK for integrating with Legalassist-AI REST API.

## Installation

```bash
pip install legalassist-sdk
# or from source
pip install -e ./sdk/python
```

## Quick Start

### Synchronous Usage

```python
from legalassist import LegalassistClient

client = LegalassistClient(api_key="your-api-key")

# Analyze document
result = client.analyze_document(
    text="Contract text here...",
    document_type="contract"
)
print(f"Job ID: {result['job_id']}")

# Wait for result
analysis = client.wait_for_analysis(result['job_id'])
print(analysis['summary'])
```

### Asynchronous Usage

```python
import asyncio
from legalassist import AsyncLegalassistClient

async def main():
    async with AsyncLegalassistClient(api_key="your-api-key") as client:
        result = await client.analyze_document(text="...")
        analysis = await client.wait_for_analysis(result['job_id'])
        print(analysis)

asyncio.run(main())
```

## Features

- ✅ Sync and async APIs
- ✅ Automatic retries
- ✅ Connection pooling
- ✅ Type hints
- ✅ Comprehensive error handling
- ✅ WebSocket support
- ✅ Rate limit handling
- ✅ Full test coverage

## Examples

### Document Analysis

```python
# Analyze from text
result = client.analyze_document(
    text="Full contract text...",
    document_type="contract",
    extract_remedies=True,
    extract_deadlines=True
)

# Or from file URL
result = client.analyze_document(
    file_url="https://example.com/contract.pdf",
    document_type="contract"
)

# Wait for completion
analysis = client.wait_for_analysis(result['job_id'], timeout=300)

print(f"Summary: {analysis['summary']}")
print(f"Remedies: {analysis['remedies']}")
print(f"Deadlines: {analysis['deadlines']}")
```

### Case Search

```python
results = client.search_cases(
    keywords=["breach of contract", "damages"],
    jurisdiction="US",
    case_type="civil",
    limit=20
)

for case in results['results']:
    print(f"{case['case_number']}: {case['title']}")
    print(f"  Score: {case['relevance_score']}")
```

### Real-time Progress Tracking (WebSocket)

```python
import websockets
import json
import asyncio

async def track_progress(job_id, token):
    url = f"ws://localhost:8000/ws/progress/{job_id}"
    
    # Pass token securely via Sec-WebSocket-Protocol
    async with websockets.connect(url, subprotocols=["access_token", token]) as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"Progress: {data.get('progress')}% - {data.get('status')}")
            if data.get('status') == 'completed':
                break
```

### Report Generation

```python
# Generate report
report = client.generate_report(
    case_id="case_123",
    report_type="comprehensive",
    format="pdf",
    include_remedies=True
)

# Wait and get download link
download_url = client.wait_for_report(report['report_id'])
print(f"Download: {download_url}")
```

### Deadlines

```python
# Get upcoming deadlines
deadlines = client.get_upcoming_deadlines(days=30)

for deadline in deadlines['deadlines']:
    if deadline['priority'] == 'critical':
        print(f"🔴 {deadline['title']} - {deadline['due_date']}")

# Create new deadline
new_deadline = client.create_deadline(
    title="Appeal Filing",
    due_date="2024-06-08T00:00:00Z",
    description="File appeal before statute of limitations",
    priority="high"
)
```

### Cost Analytics

```python
costs = client.get_cost_breakdown(period="monthly")

print(f"Total: ${costs['cost_breakdown']['total_cost']}")
print(f"LLM: ${costs['cost_breakdown']['llm_api_cost']}")
print(f"Documents: {costs['cost_breakdown']['documents_analyzed']}")
```

## Authentication

### API Key

```python
client = LegalassistClient(api_key="sk_live_...")
```

### JWT Token

```python
# Get token first
token_response = client.get_token("user@example.com", "password")
access_token = token_response['access_token']

# Use token
client = LegalassistClient(token=access_token)
```

### Create API Key

```python
# Get token
token = client.get_token("user@example.com", "password")['access_token']
client = LegalassistClient(token=token)

# Create permanent API key
api_key = client.create_api_key(
    name="Production",
    expires_in_days=90
)
print(api_key['key'])  # Save this!
```

## Advanced

### Custom Timeout

```python
client = LegalassistClient(
    api_key="your-key",
    timeout=60.0  # 60 seconds
)
```

### Error Handling

```python
from legalassist import LegalassistClient
import httpx

client = LegalassistClient(api_key="your-key")

try:
    result = client.analyze_document(text="...")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        print("Rate limited!")
    elif e.response.status_code == 401:
        print("Unauthorized!")
    else:
        print(f"Error: {e}")
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def analyze_with_retry(client, text):
    return client.analyze_document(text=text)

result = analyze_with_retry(client, "Contract...")
```

## API Reference

### LegalassistClient

#### Methods

- `get_token(username, password)` - Get access token
- `create_api_key(name, expires_in_days)` - Create API key
- `get_current_user()` - Get user info
- `analyze_document(text, file_url, document_type)` - Analyze document
- `get_analysis_status(job_id)` - Check status
- `get_analysis_result(job_id)` - Get result
- `wait_for_analysis(job_id, timeout, poll_interval)` - Wait for completion
- `search_cases(keywords, jurisdiction, limit)` - Search cases
- `get_case_timeline(case_id)` - Get case timeline
- `generate_report(case_id, report_type, format)` - Generate report
- `get_report_status(report_id)` - Check report status
- `wait_for_report(report_id, timeout)` - Wait for report
- `get_upcoming_deadlines(days)` - Get deadlines
- `create_deadline(title, due_date, description, priority)` - Create deadline
- `get_cost_breakdown(period)` - Get costs
- `get_analytics_overview()` - Get analytics

## Support

- **Issue Tracker**: [GitHub Issues](https://github.com/legalassist-ai/sdk-python)
- **Documentation**: [API Docs](/docs)
- **Email**: sdk-support@example.com
