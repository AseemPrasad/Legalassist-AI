
import sys
import os

# Add the current directory to sys.path to import core
sys.path.append(os.getcwd())

from core.app_utils import english_leakage_detected

test_cases = [
    {
        "name": "Pure Hindi",
        "text": "यह एक कानूनी सारांश है। अदालत ने फैसला सुनाया कि याचिकाकर्ता को राहत मिलनी चाहिए।",
        "expected": False
    },
    {
        "name": "Hindi with some legal terms (False Positive Risk)",
        "text": "अदालत ने Judgment सुनाया। Case Number 123 में Appeal को स्वीकार किया गया है।",
        "expected": False
    },
    {
        "name": "Pure English (Actual Leakage)",
        "text": "The court dismissed the petition. The petitioner failed to provide enough evidence. Judgment is final.",
        "expected": True
    },
    {
        "name": "Short snippet English",
        "text": "This is English leakage.",
        "expected": True
    }
]

for case in test_cases:
    result = english_leakage_detected(case["text"])
    print(f"Test: {case['name']}")
    print(f"Result: {result}")
    print(f"Match Expected: {result == case['expected']}")
    print("-" * 20)
