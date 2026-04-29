"""Shared test fixtures for CLAPnq RAG."""

import pytest


@pytest.fixture
def sample_answerable_record():
    """Sample CLAPnq answerable record."""
    return {
        "id": "test_answerable_1",
        "input": "What is the capital of France?",
        "passages": [
            {
                "title": "France",
                "text": "France is a country in Western Europe. The capital is Paris. Paris is known for its art and culture.",
                "sentences": [
                    "France is a country in Western Europe.",
                    "The capital is Paris.",
                    "Paris is known for its art and culture."
                ]
            }
        ],
        "output": [
            {
                "answer": "Paris",
                "selected_sentences": [1],
                "meta": {"skip": False}
            }
        ]
    }


@pytest.fixture
def sample_unanswerable_record():
    """Sample CLAPnq unanswerable record."""
    return {
        "id": "test_unanswerable_1",
        "input": "What is the color of the Eiffel Tower?",
        "passages": [
            {
                "title": "France",
                "text": "France is a country. It has many landmarks. The Louvre is a famous museum.",
                "sentences": [
                    "France is a country.",
                    "It has many landmarks.",
                    "The Louvre is a famous museum."
                ]
            }
        ],
        "output": [
            {
                "answer": "",
                "selected_sentences": [],
                "meta": {"skip": False}
            }
        ]
    }


@pytest.fixture
def sample_records(sample_answerable_record, sample_unanswerable_record):
    """Sample dataset with both answerable and unanswerable."""
    return [sample_answerable_record, sample_unanswerable_record]
