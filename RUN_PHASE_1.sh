#!/bin/bash
# Phase 1.1 + Phase 1.2 Execution Script with Timing

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Set Python path to include project root
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Timing functions
SCRIPT_START=$(date +%s)

log_time() {
    local step_name=$1
    local step_start=$2
    local step_end=$(date +%s)
    local duration=$((step_end - step_start))

    if [ $duration -lt 60 ]; then
        echo "⏱️  ${step_name}: ${duration}s"
    else
        local mins=$((duration / 60))
        local secs=$((duration % 60))
        echo "⏱️  ${step_name}: ${mins}m ${secs}s"
    fi
}

echo "=========================================="
echo "CLAPnq RAG System - Phase 1.1 & 1.2"
echo "=========================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Python path: ${PYTHONPATH}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON=python3
echo "Using: $($PYTHON --version)"
echo ""

# Install dependencies
echo "Step 1: Installing dependencies..."
if $PYTHON -m pip --version &> /dev/null; then
    $PYTHON -m pip install --break-system-packages -q -r requirements.txt 2>&1 | grep -i "successfully\|error" || echo "✓ Dependencies ready"
else
    echo "⚠️  Could not install with pip. Trying with existing packages..."
fi
echo ""

# Phase 1.1: Load and Chunk
echo "=========================================="
echo "Phase 1.1: Load & Chunk CLAPnq Data"
echo "=========================================="
PHASE_1_1_START=$(date +%s)

$PYTHON scripts/1_load_and_chunk.py \
    --data-dir ./data \
    --output-dir ./data/processed \
    --output-file chunks.jsonl \
    --max-tokens 512 \
    --min-tokens 50 \
    --overlap-sentences 1

PHASE_1_1_END=$(date +%s)
PHASE_1_1_DURATION=$((PHASE_1_1_END - PHASE_1_1_START))

echo ""
log_time "Phase 1.1 (Load & Chunk)" $PHASE_1_1_START
echo ""

# Check if chunks were created
if [ ! -f "./data/processed/chunks.jsonl" ]; then
    echo "❌ Chunks file not created. Aborting Phase 1.2"
    exit 1
fi

# Phase 1.2: Build Vector Store
echo "=========================================="
echo "Phase 1.2: Build Vector Store (FAISS + SQLite)"
echo "=========================================="
PHASE_1_2_START=$(date +%s)

$PYTHON scripts/2_build_vector_store.py \
    --chunks-file ./data/processed/chunks.jsonl \
    --db-path ./data/vectordb/chunks.db \
    --index-path ./data/vectordb/chunks.faiss \
    --embedding-model all-MiniLM-L6-v2 \
    --batch-size 32

PHASE_1_2_END=$(date +%s)
PHASE_1_2_DURATION=$((PHASE_1_2_END - PHASE_1_2_START))

echo ""
log_time "Phase 1.2 (Vector Store)" $PHASE_1_2_START
echo ""

# Verify Phase 1.2
echo "=========================================="
echo "Verification: Testing Vector Store"
echo "=========================================="
VERIFY_START=$(date +%s)

$PYTHON scripts/verify_phase_1_2.py

VERIFY_END=$(date +%s)
VERIFY_DURATION=$((VERIFY_END - VERIFY_START))

echo ""
log_time "Verification" $VERIFY_START

# Calculate total time
SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "=========================================="
echo "✓ Phase 1.1 & 1.2 Complete!"
echo "=========================================="
echo ""
echo "⏱️  Timing Summary:"
if [ $PHASE_1_1_DURATION -lt 60 ]; then
    echo "   Phase 1.1: ${PHASE_1_1_DURATION}s"
else
    local m=$((PHASE_1_1_DURATION / 60))
    local s=$((PHASE_1_1_DURATION % 60))
    echo "   Phase 1.1: ${m}m ${s}s"
fi

if [ $PHASE_1_2_DURATION -lt 60 ]; then
    echo "   Phase 1.2: ${PHASE_1_2_DURATION}s"
else
    local m=$((PHASE_1_2_DURATION / 60))
    local s=$((PHASE_1_2_DURATION % 60))
    echo "   Phase 1.2: ${m}m ${s}s"
fi

if [ $VERIFY_DURATION -lt 60 ]; then
    echo "   Verification: ${VERIFY_DURATION}s"
else
    local m=$((VERIFY_DURATION / 60))
    local s=$((VERIFY_DURATION % 60))
    echo "   Verification: ${m}m ${s}s"
fi

if [ $TOTAL_DURATION -lt 60 ]; then
    echo "   TOTAL: ${TOTAL_DURATION}s"
else
    local m=$((TOTAL_DURATION / 60))
    local s=$((TOTAL_DURATION % 60))
    echo "   TOTAL: ${m}m ${s}s"
fi

echo ""
echo "Output files:"
echo "  - ./data/processed/chunks.jsonl"
echo "  - ./data/vectordb/chunks.faiss"
echo "  - ./data/vectordb/chunks.db"
echo ""
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Next: Phase 1.3 - Evaluation Metrics"
