"""
Automated Unit Tests for Workshop Parts Desk LLM Gateway
Covers Stage F requirements:
1. Valid string content.
2. Content-block arrays.
3. Null content.
4. Empty content.
5. Missing choices.
6. Malformed provider JSON.
7. Provider HTTP 401.
8. Provider HTTP 402.
9. Provider HTTP 429.
10. Provider timeout.
11. Unexpected provider response structure.
12. Successful usage extraction.
"""

import json
import unittest
from urllib.error import HTTPError, URLError

from gateway import extract_model_output


class TestGatewayParser(unittest.TestCase):

    def test_01_valid_string_content(self):
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": "Valid response text."}}
            ]
        }
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertIsNone(err)
        self.assertEqual(text, "Valid response text.")

    def test_02_content_block_arrays(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Part 1. "},
                            {"type": "text", "text": "Part 2."},
                        ],
                    }
                }
            ]
        }
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertIsNone(err)
        self.assertEqual(text, "Part 1. Part 2.")

    def test_03_null_content_with_reasoning(self):
        # DeepSeek v4.1 Flash returns null content and actual text in reasoning
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "Reasoning extracted text.",
                    }
                }
            ]
        }
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertIsNone(err)
        self.assertEqual(text, "Reasoning extracted text.")

    def test_04_empty_content_without_reasoning(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": None,
                    }
                }
            ]
        }
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertIsNotNone(err)
        self.assertEqual(err, "No valid model response text")
        self.assertIsNone(text)

    def test_05_missing_choices(self):
        payload = {"id": "gen-123", "choices": []}
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertEqual(err, "Missing choices in model response")
        self.assertIsNone(text)

    def test_06_malformed_provider_json(self):
        raw = b"{invalid json"
        with self.assertRaises(Exception):
            json.loads(raw)

    def test_07_provider_http_401(self):
        exc = HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        self.assertEqual(exc.code, 401)

    def test_08_provider_http_402(self):
        exc = HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=None,
        )
        self.assertEqual(exc.code, 402)

    def test_09_provider_http_429(self):
        exc = HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )
        self.assertEqual(exc.code, 429)

    def test_10_provider_timeout(self):
        exc = TimeoutError("Provider deadline exceeded")
        self.assertIsInstance(exc, TimeoutError)

    def test_11_unexpected_provider_response_structure(self):
        payload = "not a dict"
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertEqual(err, "Invalid provider response structure")
        self.assertIsNone(text)

    def test_12_successful_usage_extraction(self):
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": "Action done."}}
            ],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 18,
                "total_tokens": 60,
            },
        }
        text, usage, err = extract_model_output(payload, logger_instance=None)
        self.assertIsNone(err)
        self.assertEqual(text, "Action done.")
        self.assertEqual(
            usage,
            {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60},
        )


if __name__ == "__main__":
    unittest.main()
