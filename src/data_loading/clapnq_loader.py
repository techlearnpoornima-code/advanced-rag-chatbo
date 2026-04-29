"""CLAPnq dataset loader."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger


class CLAPnqLoader:
    """Load and parse CLAPnq (Natural Questions) dataset from JSONL files."""

    def __init__(self):
        """Initialize loader."""
        self.records_loaded = 0
        self.validation_errors = 0

    async def load_answerable(
        self,
        filepath: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load answerable records from CLAPnq dataset.

        Args:
            filepath: Path to clapnq_train_answerable.jsonl
            limit: Maximum records to load (None = all)

        Returns:
            List of record dicts with questions and answers

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If records are malformed
        """
        logger.info("Loading answerable CLAPnq records from {}", filepath)
        return await self._load_jsonl(filepath, limit, answerable=True)

    async def load_unanswerable(
        self,
        filepath: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load unanswerable records from CLAPnq dataset.

        Args:
            filepath: Path to clapnq_train_unanswerable.jsonl
            limit: Maximum records to load (None = all)

        Returns:
            List of record dicts (with empty answers)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If records are malformed
        """
        logger.info("Loading unanswerable CLAPnq records from {}", filepath)
        return await self._load_jsonl(filepath, limit, answerable=False)

    async def _load_jsonl(
        self,
        filepath: str,
        limit: Optional[int],
        answerable: bool
    ) -> List[Dict[str, Any]]:
        """Load JSONL file with validation."""
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            records = []
            self.records_loaded = 0
            self.validation_errors = 0

            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if limit and self.records_loaded >= limit:
                        break

                    try:
                        record = json.loads(line.strip())
                        if self._validate_record(record):
                            records.append(record)
                            self.records_loaded += 1
                        else:
                            self.validation_errors += 1

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Invalid JSON at line {}: {}",
                            line_num, str(e)[:100]
                        )
                        self.validation_errors += 1

            logger.info(
                "Loaded {} records from {} (errors: {})",
                self.records_loaded, path.name, self.validation_errors
            )
            return records

        except Exception as e:
            logger.error("Error loading JSONL file {}: {}", filepath, e)
            raise

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate CLAPnq record structure.

        Expected structure:
        {
            "id": str,
            "input": str (question),
            "passages": [{"title": str, "text": str, "sentences": [str]}],
            "output": [{"answer": str, "selected_sentences": [int], "meta": dict}]
        }
        """
        try:
            # Check required fields
            if not all(k in record for k in ['id', 'input', 'passages', 'output']):
                return False

            # Validate input
            if not isinstance(record['input'], str) or len(record['input'].strip()) == 0:
                return False

            # Validate passages
            passages = record.get('passages', [])
            if not isinstance(passages, list) or len(passages) == 0:
                return False

            for passage in passages:
                required_passage_keys = ['title', 'text', 'sentences']
                if not all(k in passage for k in required_passage_keys):
                    return False

                if not isinstance(passage['text'], str):
                    return False

                if not isinstance(passage['sentences'], list):
                    return False

            # Validate output
            outputs = record.get('output', [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                return False

            for output in outputs:
                if not isinstance(output.get('answer'), str):
                    return False
                if not isinstance(output.get('selected_sentences'), list):
                    return False

            return True

        except Exception as e:
            logger.debug("Validation error: {}", e)
            return False

    def get_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about loaded records.

        Returns:
            Dict with:
            - total_records: Number of records
            - total_passages: Total passages across records
            - question_stats: Min/max/avg question length
            - passage_stats: Min/max/avg passage length
            - answer_stats: Min/max/avg answer length
        """
        if not records:
            return {}

        question_lengths = []
        passage_lengths = []
        answer_lengths = []
        answerable_count = 0

        for record in records:
            # Question length
            question = record.get('input', '')
            question_lengths.append(len(question.split()))

            # Passages
            passages = record.get('passages', [])
            for passage in passages:
                text = passage.get('text', '')
                passage_lengths.append(len(text.split()))

            # Answer length
            outputs = record.get('output', [])
            for output in outputs:
                answer = output.get('answer', '')
                answer_lengths.append(len(answer.split()) if answer.strip() else 0)
                if answer.strip():
                    answerable_count += 1

        return {
            'total_records': len(records),
            'total_passages': len(passages) * len(records),
            'answerable_rate': answerable_count / len(answer_lengths) if answer_lengths else 0,
            'question_stats': {
                'min': min(question_lengths),
                'max': max(question_lengths),
                'avg': sum(question_lengths) / len(question_lengths)
            },
            'passage_stats': {
                'min': min(passage_lengths),
                'max': max(passage_lengths),
                'avg': sum(passage_lengths) / len(passage_lengths)
            },
            'answer_stats': {
                'min': min(answer_lengths),
                'max': max(answer_lengths),
                'avg': sum(answer_lengths) / len(answer_lengths)
            }
        }
