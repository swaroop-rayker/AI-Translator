import unittest
import os
import sys

# Ensure root in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from backend.app.core.state_manager import StateManager, StateMachineError
from dataset_service.app.validation.validator import DatasetValidator
from dataset_service.app.cleaning.pipeline import DatasetCleaner

class TestPlatformCore(unittest.TestCase):
    def test_state_transitions(self):
        """
        Verify valid and invalid state transitions for dataset, training, and model.
        """
        # Test valid transition validation (should not raise exception)
        StateManager.validate_transition("dataset", "Uploaded", "Validated")
        StateManager.validate_transition("training", "Queued", "Starting")
        StateManager.validate_transition("model", "Created", "Training")

        # Test invalid transitions (should raise StateMachineError)
        with self.assertRaises(StateMachineError):
            StateManager.validate_transition("dataset", "Uploaded", "TrainReady")
        with self.assertRaises(StateMachineError):
            StateManager.validate_transition("training", "Queued", "Completed")
        with self.assertRaises(StateMachineError):
            StateManager.validate_transition("model", "Created", "Approved")

    def test_language_detection(self):
        """
        Verify character-range language detection for English, Kannada, Malayalam.
        """
        self.assertEqual(DatasetValidator.detect_lang("Hello, how are you?"), "en")
        self.assertEqual(DatasetValidator.detect_lang("ನಿಮ್ಮ ಹೆಸರು ಏನು?"), "kn")
        self.assertEqual(DatasetValidator.detect_lang("നിങ്ങളുടെ പേര് എന്താണ്?"), "ml")
        self.assertEqual(DatasetValidator.detect_lang(""), "unknown")

    def test_dataset_cleaning(self):
        """
        Verify whitespace collapsing and Unicode normalization.
        """
        raw_text = "   Hello    World!   \u00A0 " # Non-breaking space
        clean_text = DatasetCleaner.normalize_text(raw_text)
        self.assertEqual(clean_text, "Hello World!")
        
        # Test single record cleaning
        src, tgt = DatasetCleaner.clean_record("   Hello   ", "   ಹಲೋ   ")
        self.assertEqual(src, "Hello")
        self.assertEqual(tgt, "ಹಲೋ")

    def test_moses_merging(self):
        """
        Verify line-by-line Moses parallel file merging.
        """
        import tempfile
        from merge_moses import merge_moses
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "src.txt")
            tgt_file = os.path.join(tmpdir, "tgt.txt")
            out_csv = os.path.join(tmpdir, "out.csv")
            
            with open(src_file, "w", encoding="utf-8") as f:
                f.write("Hello\nWorld\n")
            with open(tgt_file, "w", encoding="utf-8") as f:
                f.write("ಹಲೋ\nಪ್ರಪಂಚ\n")
                
            merge_moses(src_file, tgt_file, out_csv)
            
            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(lines[0].strip(), "src,tgt")
                self.assertEqual(lines[1].strip(), "Hello,ಹಲೋ")
                self.assertEqual(lines[2].strip(), "World,ಪ್ರಪಂಚ")

if __name__ == "__main__":
    unittest.main()
