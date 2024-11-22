import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from cli import (
    generate_query_embedding,
    generate_text_embeddings,
    load_text_embeddings,
    chunk,
    embed,
    load,
    query,
    chat,
    get,
)

# Dummy input for functions
dummy_query = "test query"
dummy_chunks = ["chunk1", "chunk2", "chunk3"]
dummy_dimensionality = 256
dummy_batch_size = 2
dummy_df = pd.DataFrame({"book": ["dummy_book"], "chunk": ["dummy_chunk"], "embedding": [[0.1, 0.2, 0.3]]})
dummy_method = "char-split"


def test_generate_query_embedding():
    try:
        generate_query_embedding(dummy_query)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


def test_generate_text_embeddings():
    try:
        generate_text_embeddings(dummy_chunks, dimensionality=dummy_dimensionality, batch_size=dummy_batch_size)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


def test_load_text_embeddings():
    dummy_collection = MagicMock()
    try:
        load_text_embeddings(dummy_df, dummy_collection, batch_size=dummy_batch_size)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_chunk(mock_http_client):
    try:
        chunk(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_embed(mock_http_client):
    try:
        embed(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_load(mock_http_client):
    mock_client = MagicMock()
    mock_http_client.return_value = mock_client
    mock_client.create_collection.return_value = MagicMock(name="TestCollection")

    try:
        load(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_query(mock_http_client):
    mock_client = MagicMock()
    mock_http_client.return_value = mock_client
    mock_client.get_collection.return_value = MagicMock()

    try:
        query(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_chat(mock_http_client):
    mock_client = MagicMock()
    mock_http_client.return_value = mock_client
    mock_client.get_collection.return_value = MagicMock()

    try:
        chat(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")


@patch("chromadb.HttpClient")
def test_get(mock_http_client):
    mock_client = MagicMock()
    mock_http_client.return_value = mock_client
    mock_client.get_collection.return_value = MagicMock()

    try:
        get(method=dummy_method)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")

