"""Tests for CLAPnq data loader."""

import pytest
from src.data_loading.clapnq_loader import CLAPnqLoader


class TestCLAPnqLoader:
    """Tests for CLAPnqLoader."""

    @pytest.fixture
    def loader(self):
        return CLAPnqLoader()

    def test_validate_good_record(self, loader, sample_answerable_record):
        """Valid record should pass validation."""
        assert loader._validate_record(sample_answerable_record)

    def test_validate_missing_id(self, loader, sample_answerable_record):
        """Record without id should fail."""
        record = sample_answerable_record.copy()
        del record['id']
        assert not loader._validate_record(record)

    def test_validate_empty_input(self, loader, sample_answerable_record):
        """Record with empty input should fail."""
        record = sample_answerable_record.copy()
        record['input'] = ""
        assert not loader._validate_record(record)

    def test_validate_no_passages(self, loader, sample_answerable_record):
        """Record without passages should fail."""
        record = sample_answerable_record.copy()
        record['passages'] = []
        assert not loader._validate_record(record)

    def test_validate_missing_passage_fields(self, loader, sample_answerable_record):
        """Passage missing required fields should fail."""
        record = sample_answerable_record.copy()
        del record['passages'][0]['sentences']
        assert not loader._validate_record(record)

    def test_statistics_calculation(self, loader, sample_records):
        """Statistics should be calculated correctly."""
        stats = loader.get_statistics(sample_records)

        assert 'total_records' in stats
        assert stats['total_records'] == 2
        assert 'question_stats' in stats
        assert 'passage_stats' in stats
        assert 'answer_stats' in stats

        # Check structure
        assert 'min' in stats['question_stats']
        assert 'max' in stats['question_stats']
        assert 'avg' in stats['question_stats']
