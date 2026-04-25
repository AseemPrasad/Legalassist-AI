# Contributing Guide

## Regional Legal Help Directory

The app reads local legal help data from [legal_aid_directory.json](legal_aid_directory.json).

### Data format

The file contains:

- `metadata`: versioning and source notes
- `states`: object with one key per state/UT

Each state/UT must include:

```json
{
  "legal_aid_authority": {
    "name": "...",
    "phone": "...",
    "website": "..."
  },
  "law_colleges": [
    {
      "name": "...",
      "city": "...",
      "clinic_available": true
    }
  ],
  "ngos": [
    {
      "name": "...",
      "specialty": "...",
      "phone": "...",
      "website": "..."
    }
  ],
  "bar_association": {
    "name": "...",
    "phone": "...",
    "website": "..."
  },
  "avg_cost": "INR 5000-15000 for appeal"
}
```

## How to add a new state or UT entry

1. Open [legal_aid_directory.json](legal_aid_directory.json).
2. Add a new object under `states` using the exact schema above.
3. Keep the state/UT name consistent with official naming.
4. Add at least:
   - One legal aid authority contact
   - One law college
   - One NGO
   - One bar association contact
   - One `avg_cost` estimate
5. Update `metadata.last_updated` with the current date.

## Data quality checklist

Before committing directory updates:

1. Validate JSON syntax.
2. Confirm authority and bar websites open in a browser.
3. Confirm phone numbers are reachable/publicly listed.
4. Confirm NGO websites are active.
5. Ensure at least one law college includes `clinic_available: true` where applicable.

## Optional quick JSON validation

Run this command from the project root:

```bash
python -m json.tool legal_aid_directory.json > NUL
```

On success, the command exits without error.
