import unittest
from unittest.mock import patch
import io
import sys # Required for patching sys.stdout

from src.confirmations import request_user_confirmation

class TestRequestUserConfirmation(unittest.TestCase):

    @patch('builtins.input')
    def test_confirm_yes_variations(self, mock_input):
        prompt = "Confirm?"
        yes_inputs = ['y', 'Y', ' y ', ' Y ']
        for val in yes_inputs:
            mock_input.return_value = val
            with self.subTest(input_value=val):
                self.assertTrue(request_user_confirmation(prompt))

    @patch('builtins.input')
    def test_confirm_no_variations(self, mock_input):
        prompt = "Confirm?"
        no_inputs = ['n', 'N', ' n ', '', 'no', 'random', 'yes'] # 'yes' is not 'y'
        for val in no_inputs:
            mock_input.return_value = val
            with self.subTest(input_value=val):
                self.assertFalse(request_user_confirmation(prompt))

    @patch('builtins.input', return_value='y') # Mock input to prevent hanging
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_prompt_message_display(self, mock_stdout, mock_input):
        prompt = "Is this the correct prompt?"
        request_user_confirmation(prompt)
        # The prompt in request_user_confirmation adds a space at the end
        self.assertEqual(mock_stdout.getvalue(), prompt + " ")

    @patch('builtins.input', side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_handling(self, mock_input):
        prompt = "Interrupt test"
        # Check that the function returns False and logs the correct message
        with self.assertLogs(level='INFO') as log_cm:
            self.assertFalse(request_user_confirmation(prompt))
        # Check for the specific log message
        # The logged message in confirmations.py starts with '\n'
        self.assertTrue(any("\nConfirmation cancelled by user." in message for message in log_cm.output))

    @patch('builtins.input', side_effect=EOFError)
    def test_eof_error_handling(self, mock_input):
        prompt = "EOF test"
        # Check that the function returns False and logs the correct message
        with self.assertLogs(level='INFO') as log_cm:
            self.assertFalse(request_user_confirmation(prompt))
        # Check for the specific log message
        # The logged message in confirmations.py starts with '\n'
        self.assertTrue(any("\nConfirmation input stream closed." in message for message in log_cm.output))

if __name__ == '__main__':
    unittest.main()
