# GitHub Issue: Fix Bare Exception Handlers

**Title:** 🐛 Replace bare `except:` clauses with specific exception handling

**Description:**

## Problem
The codebase contains 3 instances of bare `except:` statements that catch all exceptions indiscriminately, including `KeyboardInterrupt` and `SystemExit`. This violates Python best practices and makes debugging harder.

## Files Affected
1. `app.py` (line 343)
2. `pdf_exporter.py` (line 162)
3. `report_service.py` (line 82)

## Changes Made
- ✅ `app.py`: Changed `except: pass` to `except (ValueError, TypeError): pass`
- ✅ `pdf_exporter.py`: Changed `except: pass` to `except Exception:` with logging
- ✅ `report_service.py`: Changed `except:` to `except Exception:`

## Testing
```bash
# Run existing tests
pytest tests/

# Verify no regressions
python app.py
```

## Labels
- `type: code-quality`
- `priority: medium`
- `area: error-handling`
