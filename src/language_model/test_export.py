import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
import torch
import datetime
from ..export import export_model_as_safetensors

class TestExportModel(unittest.TestCase):
     @patch('os.makedirs')
     @patch('torch.load')
     @patch('safetensors.torch.save_file')
     @patch('shutil.copy')
     @patch('builtins.open', new_callable=mock_open)
     @patch('json.dump')
     @patch('datetime.datetime')
     def test_export_model_as_safetensors(self, mock_datetime, mock_json_dump, 
                                                    mock_file, mock_shutil_copy, 
                                                    mock_save_file, mock_torch_load, 
                                                    mock_makedirs):
          # Setup datetime mock
          mock_datetime.now.return_value = datetime.datetime(2023, 5, 15, 10, 30)
          
          # Setup file read mock for vocab
          vocab_data = {"token1": 0, "token2": 1, "token3": 2, "token4": 3, 
                           "token5": 4, "token6": 5, "token7": 6, "token8": 7, "token9": 8}
          
          # Create different file handlers for different file operations
          file_mock_instances = {
               os.path.join('data', 'output', 'vocab_subword.json'): mock_open(read_data=json.dumps(vocab_data)).return_value
          }
          mock_file.side_effect = lambda filename, *args, **kwargs: file_mock_instances.get(filename, MagicMock())
          
          # Setup model mock
          mock_model = MagicMock()
          mock_model.token_embedding_table.weight.shape = (9, 768)
          mock_model.blocks = [MagicMock(), MagicMock()]
          mock_model.blocks[0].sa.heads = [MagicMock(), MagicMock(), MagicMock()]
          
          # Mock GPTLanguageModel to return our mock model
          with patch('export.GPTLanguageModel', return_value=mock_model):
               # Call the function
               export_model_as_safetensors()
               
               # Verify expected behavior
               expected_export_path = "data/output/hf_model_5_15_10_30"
               
               # Check directory creation
               mock_makedirs.assert_called_once_with(expected_export_path, exist_ok=True)
               
               # Check model creation and loading
               self.assertTrue(mock_torch_load.called)
               mock_torch_load.assert_called_once_with("data/output/chat_aligned_best_model.pt")
               
               # Check safetensors save
               self.assertTrue(mock_save_file.called)
               model_path = os.path.join(expected_export_path, "model.safetensors")
               self.assertEqual(mock_save_file.call_args[0][1], model_path)
               
               # Check file copies
               expected_copy_calls = [
                    call("src/language_model/subword_tokenizer.py", os.path.join(expected_export_path, "tokenizer.py")),
                    call("src/language_model/gpt.py", os.path.join(expected_export_path, "model.py"))
               ]
               mock_shutil_copy.assert_has_calls(expected_copy_calls, any_order=True)
               
               # Check JSON dumps for config files
               self.assertEqual(mock_json_dump.call_count, 4)  # config.json, vocab, tokenizer_config, generation_config

     @patch('os.makedirs')
     @patch('builtins.open', new_callable=mock_open)
     def test_export_model_handles_file_error(self, mock_file, mock_makedirs):
          # Test handling of file-related errors
          mock_file.side_effect = IOError("File error")
          
          with patch('export.GPTLanguageModel') as mock_gpt:
               with self.assertRaises(IOError):
                    export_model_as_safetensors()

if __name__ == '__main__':
     unittest.main()