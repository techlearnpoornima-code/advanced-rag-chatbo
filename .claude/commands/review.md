# Code Review Command

## Command: `/review`

Perform comprehensive code review on changed files.

## What to Check

### 1. Code Style
- [ ] Follows PEP 8 guidelines
- [ ] Type hints present for all functions
- [ ] Docstrings for public methods
- [ ] Proper naming conventions (snake_case, PascalCase)
- [ ] No hardcoded values (use config)
- [ ] Async/await used for I/O operations

### 2. Error Handling
- [ ] Specific exceptions (not bare `except:`)
- [ ] Errors logged with context
- [ ] Graceful degradation
- [ ] User-friendly error messages in API

### 3. Testing
- [ ] Tests added for new features
- [ ] Edge cases covered
- [ ] Mocks used for external dependencies
- [ ] Tests pass locally

### 4. Performance
- [ ] No blocking I/O in async functions
- [ ] Database queries optimized
- [ ] Caching used where appropriate
- [ ] No N+1 query problems

### 5. Security
- [ ] No API keys in code
- [ ] Input validation present
- [ ] SQL injection prevention
- [ ] XSS prevention in responses

### 6. Documentation
- [ ] README updated if needed
- [ ] API docs updated
- [ ] Docstrings complete
- [ ] Comments for complex logic

### 7. Architecture
- [ ] Follows project structure
- [ ] Separation of concerns
- [ ] DRY principle
- [ ] Single responsibility

## Review Process

1. Read the changed files
2. Check against each category above
3. Provide specific, actionable feedback
4. Suggest improvements with code examples
5. Highlight what was done well

## Example Output

```markdown
## Code Review Results

### ✅ Strengths
- Good use of type hints throughout
- Comprehensive error handling
- Well-documented functions

### ⚠️ Issues Found

#### High Priority
1. **File: src/core/handlers.py, Line 45**
   - Issue: Using blocking I/O in async function
   - Current: `data = requests.get(url)`
   - Suggested: Use `aiohttp` instead
   ```python
   async with aiohttp.ClientSession() as session:
       async with session.get(url) as response:
           data = await response.json()
   ```

#### Medium Priority
2. **File: app/api/routes.py, Line 23**
   - Issue: Missing input validation
   - Add validation for max query length

#### Low Priority
3. **File: src/core/intent_classifier.py, Line 67**
   - Issue: Magic number in code
   - Move `0.7` to config as `MIN_CONFIDENCE_THRESHOLD`

### 📝 Suggestions
- Consider adding caching for frequently accessed data
- Could extract repeated logic into a helper function
```

## Usage

```
/review <file_path>
```

Or review all changed files:
```
/review
```
