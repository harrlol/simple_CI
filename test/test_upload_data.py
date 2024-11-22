import pytest
from unittest.mock import Mock, patch, mock_open
import os
from google.cloud import storage
from google.api_core import exceptions
import glob
from ..src.data_upload_service.upload_data import upload_to_gcp

class TestUploadData:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup environment variables for all tests"""
        os.environ['GCP_PROJECT'] = 'test-project'
        os.environ['GCS_BUCKET_NAME'] = 'test-bucket'
        yield
        # Cleanup
        del os.environ['GCP_PROJECT']
        del os.environ['GCS_BUCKET_NAME']

    @pytest.fixture
    def mock_storage_client(self):
        """Mock GCP storage client"""
        with patch('google.cloud.storage.Client') as mock_client:
            # Create mock bucket
            mock_bucket = Mock()
            mock_bucket.name = 'test-bucket'
            mock_client.return_value.get_bucket.return_value = mock_bucket
            
            # Create mock blob
            mock_blob = Mock()
            mock_blob.exists.return_value = False
            mock_bucket.blob.return_value = mock_blob
            
            yield mock_client, mock_bucket, mock_blob

    @pytest.fixture
    def mock_glob(self):
        """Mock glob file patterns"""
        with patch('glob.glob') as mock_glob:
            mock_glob.side_effect = lambda pattern: {
                '/app/data/embeddings-recursive-split-*.jsonl': [
                    '/app/data/embeddings-recursive-split-1.jsonl',
                    '/app/data/embeddings-recursive-split-2.jsonl'
                ],
                '/app/data/chunks-recursive-split-*.jsonl': [
                    '/app/data/chunks-recursive-split-1.jsonl',
                    '/app/data/chunks-recursive-split-2.jsonl'
                ]
            }[pattern]
            yield mock_glob

    def test_successful_upload(self, mock_storage_client, mock_glob):
        """Test successful upload of files to GCP"""
        mock_client, mock_bucket, mock_blob = mock_storage_client
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify storage client was initialized with correct project
        mock_client.assert_called_once_with(project='test-project')
        
        # Verify bucket was retrieved
        mock_client.return_value.get_bucket.assert_called_once_with('test-bucket')
        
        # Verify correct number of files were processed
        assert mock_bucket.blob.call_count == 4  # 2 embedding files + 2 chunk files
        
        # Verify upload_from_filename was called for each file
        assert mock_blob.upload_from_filename.call_count == 4

    def test_bucket_not_found(self, mock_storage_client, mock_glob):
        """Test handling of non-existent bucket"""
        mock_client, _, _ = mock_storage_client
        mock_client.return_value.get_bucket.side_effect = exceptions.NotFound('Bucket not found')
        mock_client.return_value.create_bucket.return_value = Mock()
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify bucket creation was attempted
        mock_client.return_value.create_bucket.assert_called_once_with(
            'test-bucket', 
            location="us-central1"
        )

    def test_insufficient_permissions(self, mock_storage_client, mock_glob):
        """Test handling of insufficient permissions"""
        mock_client, _, _ = mock_storage_client
        mock_client.return_value.get_bucket.side_effect = exceptions.NotFound('Bucket not found')
        mock_client.return_value.create_bucket.side_effect = exceptions.Forbidden('Insufficient permissions')
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify bucket creation was attempted
        mock_client.return_value.create_bucket.assert_called_once()

    def test_skip_existing_files(self, mock_storage_client, mock_glob):
        """Test skipping of existing files"""
        mock_client, mock_bucket, mock_blob = mock_storage_client
        # Set blob to exist
        mock_blob.exists.return_value = True
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify exists() was called but upload_from_filename wasn't
        assert mock_blob.exists.call_count == 4
        assert mock_blob.upload_from_filename.call_count == 0

    def test_upload_error_handling(self, mock_storage_client, mock_glob):
        """Test handling of upload errors"""
        mock_client, mock_bucket, mock_blob = mock_storage_client
        mock_blob.upload_from_filename.side_effect = Exception('Upload failed')
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify upload was attempted for all files
        assert mock_blob.upload_from_filename.call_count == 4

    @patch('builtins.print')
    def test_error_messages(self, mock_print, mock_storage_client, mock_glob):
        """Test error message printing"""
        mock_client, _, mock_blob = mock_storage_client
        mock_blob.upload_from_filename.side_effect = Exception('Upload failed')
        
        # Run the upload function
        upload_to_gcp()
        
        # Verify error messages were printed
        mock_print.assert_any_call('❌ Error uploading /app/data/embeddings-recursive-split-1.jsonl: Upload failed')

    def test_no_files_found(self, mock_storage_client):
        """Test handling when no files are found"""
        with patch('glob.glob', return_value=[]):
            # Run the upload function
            upload_to_gcp()
            
            mock_client, mock_bucket, mock_blob = mock_storage_client
            # Verify no uploads were attempted
            assert mock_blob.upload_from_filename.call_count == 0
