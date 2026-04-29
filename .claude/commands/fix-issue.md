# Fix Issue Command

## Command: `/fix-issue`

Fix a specific issue or bug in the codebase.

## Process

### 1. Understand the Issue
- Read the error message or bug description
- Check relevant logs
- Identify affected components
- Determine root cause

### 2. Plan the Fix
- Identify which files need changes
- Consider side effects
- Plan testing approach
- Check for similar issues elsewhere

### 3. Implement the Fix
- Make minimal changes to fix the issue
- Follow code style guidelines
- Add error handling if missing
- Update tests

### 4. Verify the Fix
- Run existing tests
- Add new tests for the bug
- Test edge cases
- Check logs for errors

### 5. Document
- Add comments explaining the fix
- Update documentation if needed
- Note any technical debt

## Common Issue Types

### API Errors
```python
# Issue: 500 error on /api/v1/chat
# Root cause: Unhandled exception in intent classifier

# Fix:
try:
    intent = await self.classifier.classify(query)
except Exception as e:
    logger.error(f"Intent classification failed: {e}")
    # Fallback to default intent
    intent = IntentAnalysis(
        primary_intent=QueryIntent.CONVERSATIONAL,
        confidence=0.5,
        sub_intents=None,
        is_multi_intent=False
    )
```

### Performance Issues
```python
# Issue: Slow query responses
# Root cause: Too many database queries

# Fix: Add caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embeddings(text: str) -> List[float]:
    return self.model.encode(text)
```

### Data Issues
```python
# Issue: Empty results from vector store
# Root cause: Vector DB not initialized

# Fix: Add initialization check
async def search(self, query: str) -> List[RetrievedChunk]:
    if self.collection is None:
        logger.error("Vector store not initialized")
        raise RuntimeError("Vector store not ready")
    
    # Proceed with search
    ...
```

## Usage

```bash
# Fix a specific issue
/fix-issue "500 error on chat endpoint"

# Fix with context
/fix-issue --file app/api/routes.py --line 45 "TypeError: NoneType"

# Fix and add tests
/fix-issue "Multi-intent detection failing" --add-tests
```

## Checklist

- [ ] Issue reproduced locally
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Tests updated/added
- [ ] No new warnings/errors
- [ ] Documentation updated
- [ ] Logs checked
- [ ] Edge cases considered

## Output Format

```markdown
## Issue Fixed

**Problem**: 500 error on /api/v1/chat when query is empty

**Root Cause**: Missing input validation in ChatRequest model

**Files Changed**:
- app/models.py
- tests/test_api.py

**Changes Made**:
1. Added validation in ChatRequest:
   ```python
   query: str = Field(..., min_length=1, max_length=2000)
   ```

2. Added test case:
   ```python
   def test_empty_query_validation(client):
       response = client.post("/api/v1/chat", json={"query": ""})
       assert response.status_code == 422
   ```

**Testing**:
- ✅ Unit tests pass
- ✅ API test added
- ✅ Manual testing confirmed

**Side Effects**: None

**Follow-up**: Consider adding client-side validation as well
```
