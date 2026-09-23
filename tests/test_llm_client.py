"""Exercise the installed OpenAI SDK over a local mock HTTP transport."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from llm_client import LLMError, generate_text


def response_body(text="Priya owns GBR-1234. [Role-DB]", status="completed"):
    return {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": "test-model", "status": status,
        "output": [{
            "id": "msg_test", "type": "message", "role": "assistant",
            "status": "completed", "content": [{
                "type": "output_text", "text": text, "annotations": [],
            }],
        }],
    }


class LLMClientTests(unittest.TestCase):
    def call_with_transport(self, handler, **kwargs):
        def factory(**config):
            return OpenAI(**config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-only", "OPENAI_MODEL": "test-model"}, clear=True):
            with patch("openai.OpenAI", side_effect=factory):
                return generate_text("System instructions", {"query": "test"}, **kwargs)

    def test_real_sdk_serialization_and_response_parsing(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=response_body())
        answer = self.call_with_transport(handler)
        self.assertIn("GBR-1234", answer)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/responses")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["store"])
        self.assertEqual(json.loads(body["input"]), {"query": "test"})
        self.assertEqual(body["instructions"], "System instructions")

    def test_json_schema_is_sent_to_responses_api(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["text"]["format"]["schema"], schema)
            self.assertTrue(body["text"]["format"]["strict"])
            return httpx.Response(200, json=response_body("{}"))
        self.assertEqual(self.call_with_transport(handler, schema=schema), "{}")

    def test_api_error_does_not_echo_sensitive_body(self):
        def handler(request):
            return httpx.Response(401, json={"error": {
                "message": "secret-body unit-test-only", "type": "authentication_error",
            }})
        with self.assertRaises(LLMError) as error:
            self.call_with_transport(handler)
        self.assertIn("HTTP 401", str(error.exception))
        self.assertNotIn("secret-body", str(error.exception))
        self.assertNotIn("unit-test-only", str(error.exception))

    def test_empty_and_incomplete_answers_rejected(self):
        for text, status in [("", "completed"), ("Partial answer", "incomplete")]:
            with self.subTest(status=status), self.assertRaises(LLMError):
                self.call_with_transport(lambda request: httpx.Response(200, json=response_body(text, status)))

    def test_exhausted_credits_have_specific_safe_guidance(self):
        def handler(request):
            return httpx.Response(429, headers={"x-should-retry": "false"}, json={"error": {
                "code": "credit_balance_exhausted", "type": "insufficient_quota",
                "message": "private server details unit-test-only",
            }})
        with self.assertRaises(LLMError) as error:
            self.call_with_transport(handler)
        self.assertIn("credit balance is exhausted", str(error.exception))
        self.assertIn("billing", str(error.exception))
        self.assertNotIn("private server", str(error.exception))
        self.assertNotIn("unit-test-only", str(error.exception))

    def test_temporary_rate_limit_does_not_claim_exhausted_credits(self):
        def handler(request):
            return httpx.Response(429, headers={"x-should-retry": "false"}, json={"error": {
                "code": "rate_limit_exceeded", "type": "rate_limit_error", "message": "wait",
            }})
        with self.assertRaises(LLMError) as error:
            self.call_with_transport(handler)
        self.assertIn("HTTP 429", str(error.exception))
        self.assertNotIn("credit balance is exhausted", str(error.exception))

    def test_missing_key_fails_before_client_creation(self):
        with patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI") as client:
            with self.assertRaisesRegex(LLMError, "OPENAI_API_KEY"):
                generate_text("system", {})
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
