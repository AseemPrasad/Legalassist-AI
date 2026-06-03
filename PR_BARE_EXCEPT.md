# GitHub PR: Fix Bare Exception Handlers

**Title:** 🔧 refactor: replace bare except clauses with specific exception handling

**Description:**

## Summary
This PR fixes 3 instances of bare `except:` statements that violate Python best practices and make error tracking difficult.

## Changes
- **app.py** (line 343): `except: pass` → `except (ValueError, TypeError): pass`
- **pdf_exporter.py** (line 162): `except: pass` → `except Exception:` + logging
- **report_service.py** (line 82): `except:` → `except Exception:`

## Why?
✅ Prevents catching system-level exceptions  
✅ Improves debuggability  
✅ Follows PEP 8 and Python best practices  
✅ Better error tracking and logging

## Testing
- [x] All existing tests pass
- [x] No regressions in PDF export
- [x] No regressions in S3 storage
- [x] Analytics preview still works

## Breaking Changes
❌ None

## Related Issues
N/A

---

### Checklist
- [x] Code follows style guidelines
- [x] Tests added/updated
- [x] Documentation updated (if needed)
- [x] No breaking changes
