# LegalAssist AI API Integration Guide

This guide provides API integration details and usage examples for interacting with the LegalAssist AI REST endpoints.

## Authentication

The API supports both JWT Bearer and API Key authentications.

### 1. Bearer JWT Authentication
Obtain your JWT token by sending a request to the login/auth endpoint:

```bash
curl -X POST "https://api.legalassist.ai/api/v1/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "otp": "123456"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Include it in subsequent requests as:
`-H "Authorization: Bearer <your_access_token>"`

### 2. API Key Authentication
API Keys can be configured in your settings dashboard. Include the key in your headers:
`-H "X-API-Key: <your_api_key>"`

---

## Key Endpoints

### 1. Cases Management
- **Create Case**: `POST /api/v1/cases/`
- **List Cases**: `GET /api/v1/cases/`
- **Get Case**: `GET /api/v1/cases/{case_id}`

### 2. Documents & Summaries
- **Upload Document**: `POST /api/v1/documents/upload`
- **Get Summaries**: `GET /api/v1/documents/{document_id}/summary`

### 3. Deadlines
- **Create Deadline**: `POST /api/v1/deadlines/`
- **Get Upcoming Deadlines**: `GET /api/v1/deadlines/upcoming`
