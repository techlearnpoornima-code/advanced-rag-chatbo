#!/bin/bash
# Validation hook for bash scripts
# Prevents unsafe operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running pre-commit validation...${NC}"

# Check for dangerous commands
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "dd if="
    "mkfs"
    "> /dev/"
    ":(){ :|:& };:"
)

FILES_TO_CHECK=$(find . -name "*.sh" -type f 2>/dev/null)

for file in $FILES_TO_CHECK; do
    for pattern in "${DANGEROUS_PATTERNS[@]}"; do
        if grep -q "$pattern" "$file"; then
            echo -e "${RED}ERROR: Dangerous command found in $file: $pattern${NC}"
            echo "Blocked for safety. Please review."
            exit 1
        fi
    done
done

# Check for missing .env
if [ ! -f .env ] && [ -f .env.example ]; then
    echo -e "${YELLOW}WARNING: .env file not found. Copy from .env.example${NC}"
fi

# Check for hardcoded API keys
if grep -r "sk-ant-" --include="*.py" --include="*.js" . 2>/dev/null; then
    echo -e "${RED}ERROR: Potential API key found in code!${NC}"
    echo "Never commit API keys. Use environment variables."
    exit 1
fi

echo -e "${GREEN}✓ Validation passed${NC}"
exit 0
