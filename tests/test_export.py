import unittest
from unittest.mock import patch, mock_open, MagicMock
import datetime
from language_model.scripts.export_to_safetensors import export_model_as_safetensors

class TestExportModel(unittest.TestCase):
     @patch('language_model.scripts.export_to_safetensors.save_model')
     @patch('language_model.scripts.export_to_safetensors.GPTLanguageModel')
     @patch('language_model.scripts.export_to_safetensors.shutil.copy')
     @patch('builtins.open', new_callable=mock_open)
     def test_export_model_as_safetensors(self, mock_file, mock_shutil_copy, mock_gpt_model, mock_save_model):
          """Test basic export model functionality - simplified test"""
          # Setup model mock
          mock_model = MagicMock()
          mock_model.token_embedding_table.weight.shape = (9, 768)
          mock_model.blocks = [MagicMock(), MagicMock()]
          mock_model.blocks[0].sa.heads = [MagicMock(), MagicMock(), MagicMock()]
          mock_gpt_model.return_value = mock_model
          
          # Mock json.load to return vocab data
          vocab_data = {
               "model": {
                    "vocab": {"token1": 0, "token2": 1, "token3": 2, "token4": 3, 
                              "token5": 4, "token6": 5, "token7": 6, "token8": 7, "token9": 8}
               }
          }
          
          with patch('language_model.scripts.export_to_safetensors.json.load', return_value=vocab_data):
               with patch('language_model.scripts.export_to_safetensors.torch.load', return_value={}):
                    with patch('language_model.scripts.export_to_safetensors.os.makedirs'):
                         with patch('language_model.scripts.export_to_safetensors.datetime') as mock_datetime:
                              mock_datetime.now.return_value = datetime.datetime(2023, 5, 15, 10, 30)
                              
                              # Call the function
                              export_model_as_safetensors()
                              
                              # Verify save_model was called
                              self.assertTrue(mock_save_model.called)
                              # Verify GPTLanguageModel was created with correct vocab size
                              mock_gpt_model.assert_called_once_with(vocab_size=9)
                              # Verify shutil.copy was called for copying files
                              self.assertTrue(mock_shutil_copy.called)

     @patch('os.makedirs')
     @patch('builtins.open', new_callable=mock_open)
     def test_export_model_handles_file_error(self, mock_file, mock_makedirs):
          # Test handling of file-related errors
          mock_file.side_effect = IOError("File error")
          
          with patch('language_model.scripts.export_to_safetensors.GPTLanguageModel') as mock_gpt:
               with self.assertRaises(IOError):
                    export_model_as_safetensors()

if __name__ == '__main__':
     unittest.main()