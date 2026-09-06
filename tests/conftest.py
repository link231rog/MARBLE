"""Global pytest configuration for MARBLE.

Enforces offline mode for Hugging Face and Transformers during tests to guarantee
that test suites never contact Hugging Face Hub or download model weights to local disk.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
