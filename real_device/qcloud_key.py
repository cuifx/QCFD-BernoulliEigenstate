"""
qcloud_key.py -- centralized QCloud credential loader
======================================================
Loads the Origin Quantum cloud API key without hard-coding it in every
experiment script.

Resolution order:
  1. Environment variable QCIRCUIT_API_KEY
  2. Local file qcloud_key.txt in this directory (never commit to
     version control; add to .gitignore)

Usage:
    from qcloud_key import get_api_key
    service = QCloudService(get_api_key())
"""
import os


def get_api_key():
    key = os.environ.get("QCIRCUIT_API_KEY")
    if key:
        return key.strip()

    here = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(here, "qcloud_key.txt")
    if os.path.isfile(key_file):
        with open(key_file, encoding="utf-8") as f:
            # the file may carry a header comment block; only the first
            # non-comment, non-empty line is the key itself
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s

    raise RuntimeError(
        "QCloud API key not found. Set the QCIRCUIT_API_KEY environment "
        "variable or create code/real_device/qcloud_key.txt with the key."
    )
