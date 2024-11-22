# tests/test_data_versioning.py
import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from google.cloud import storage
import vertexai
from cli import generate, prepare, upload, read_gcs_file

class TestDataVersioning:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup environment variables and mock data for all tests"""
        # Set environment variables
        os.environ["GCP_PROJECT"] = "test-project"
        os.environ["GCS_BUCKET_NAME"] = "test-bucket"
        
        # Sample test data
        self.test_medical_note = """
        Patient presents with fever and cough for 3 days.
        Vital signs: Temperature 38.5°C, BP 120/80, HR 88
        Lungs: Crackles in right lower lobe
        Assessment: Suspected pneumonia
        Plan: Order chest x-ray, start antibiotics
        """
        
        self.sample_qa_json = """[
            {"question": "What kinds of tests should a doctor order if a patient presents with fever and cough?",
             "answer": "Chest x-ray should be ordered to evaluate for pneumonia"},
            {"question": "What might the diagnosis be if a patient has crackles in right lower lobe?",
             "answer": "The diagnosis might be pneumonia based on the presence of crackles"}
        ]"""
        
        yield
        
        # Cleanup
        if "GCP_PROJECT" in os.environ:
            del os.environ["GCP_PROJECT"]
        if "GCS_BUCKET_NAME" in os.environ:
            del os.environ["GCS_BUCKET_NAME"]

    @pytest.fixture
    def mock_gcs(self):
        """Mock Google Cloud Storage"""
        with patch('google.cloud.storage.Client') as mock_client:
            # Setup mock bucket and blob
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.name = 'test/clinical_note_1.txt'
            mock_blob.download_as_text.return_value = self.test_medical_note
            mock_bucket.list_blobs.return_value = [mock_blob]
            mock_client.return_value.get_bucket.return_value = mock_bucket
            yield mock_client

    @pytest.fixture
    def mock_vertexai(self):
        """Mock Vertex AI"""
        with patch('vertexai.init') as mock_init:
            with patch('vertexai.generative_models.GenerativeModel') as mock_model:
                mock_response = MagicMock()
                mock_response.text = self.sample_qa_json
                mock_model.return_value.generate_content.return_value = mock_response
                yield mock_model

    def test_read_gcs_file(self, mock_gcs):
        """Test reading files from GCS bucket"""
        blobs = read_gcs_file('test-bucket', 'test/')
        
        # Verify GCS client was initialized
        mock_gcs.assert_called_once()
        
        # Verify bucket operations
        mock_gcs.return_value.get_bucket.assert_called_once_with('test-bucket')
        mock_gcs.return_value.get_bucket.return_value.list_blobs.assert_called_once()

    def test_generate_success(self, mock_gcs, mock_vertexai, tmp_path):
        """Test successful generation of Q&A pairs"""
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open', mock_open()) as mock_file:
                generate()
                
                # Verify directory creation
                mock_makedirs.assert_called_once()
                
                # Verify Vertex AI initialization
                mock_vertexai.assert_called_once()
                
                # Verify file operations
                mock_file.assert_called()
                
                # Verify content generation
                mock_vertexai.return_value.generate_content.assert_called_once()

    def test_generate_error_handling(self, mock_gcs, mock_vertexai):
        """Test error handling in generate function"""
        # Simulate Vertex AI error
        mock_vertexai.return_value.generate_content.side_effect = Exception("API Error")
        
        with patch('os.makedirs'), patch('builtins.open', mock_open()):
            with pytest.raises(Exception) as exc_info:
                generate()
            assert "API Error" in str(exc_info.value)

    def test_prepare_valid_data(self, tmp_path):
        """Test data preparation with valid input"""
        # Create test directory and files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create test JSON file
        with open(data_dir / "out_1.json", "w") as f:
            f.write(self.sample_qa_json)
        
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = [str(data_dir / "out_1.json")]
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                prepare()
                mock_to_csv.assert_called()

    def test_prepare_invalid_data(self, tmp_path):
        """Test data preparation with invalid input"""
        # Create test directory and files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create invalid JSON file
        with open(data_dir / "out_1.json", "w") as f:
            f.write('{"invalid": "json"')
        
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = [str(data_dir / "out_1.json")]
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                prepare()
                # Verify error handling
                mock_to_csv.assert_not_called()

    def test_upload_to_gcs(self, mock_gcs):
        """Test uploading files to GCS"""
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = ['data/test.jsonl', 'data/test.csv']
            upload()
            
            # Verify GCS operations
            mock_gcs.assert_called_once()
            bucket = mock_gcs.return_value.get_bucket.return_value
            assert bucket.blob.call_count == 2

    def test_full_pipeline_integration(self, mock_gcs, mock_vertexai, tmp_path):
        """Test the entire pipeline integration"""
        with patch('os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('glob.glob') as mock_glob, \
             patch('pandas.DataFrame.to_csv'):
            
            # Setup mock for file operations
            mock_glob.return_value = ['data/test.jsonl', 'data/test.csv']
            
            # Run full pipeline
            generate()
            prepare()
            upload()
            
            # Verify all major components were called
            mock_vertexai.assert_called_once()
            mock_gcs.assert_called()
            assert mock_glob.call_count > 0

    def test_data_validation(self):
        """Test data validation functions"""
        # Test valid Q&A format
        valid_qa = json.loads(self.sample_qa_json)
        assert all('question' in qa and 'answer' in qa for qa in valid_qa)
        
        # Test data structure
        assert isinstance(valid_qa, list)
        assert len(valid_qa) > 0
        assert all(isinstance(qa, dict) for qa in valid_qa)