# -*- coding: utf-8 -*-
"""Unit tests for MiniMaxClient integration."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path so we can import without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMiniMaxClientInit(unittest.TestCase):
    """Test MiniMaxClient initialization."""

    def test_default_initialization(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False):
            client = MiniMaxClient()
            self.assertEqual(client.api_key, "test-key")
            self.assertEqual(client.model_name, "MiniMax-M2.7")
            self.assertEqual(client.base_url, "https://api.minimax.io/v1")

    def test_custom_initialization(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        client = MiniMaxClient(api_key="custom-key", model_name="MiniMax-M2.5")
        self.assertEqual(client.api_key, "custom-key")
        self.assertEqual(client.model_name, "MiniMax-M2.5")

    def test_env_var_fallback(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {}, clear=True):
            client = MiniMaxClient()
            self.assertEqual(client.api_key, "")
            self.assertEqual(client.model_name, "MiniMax-M2.7")

    def test_explicit_api_key_overrides_env(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False):
            client = MiniMaxClient(api_key="explicit-key")
            self.assertEqual(client.api_key, "explicit-key")


class TestMiniMaxClientTemperatureClamping(unittest.TestCase):
    """Test that temperature is clamped to [0.0, 1.0]."""

    @patch("openai.OpenAI")
    def test_temperature_clamped_high(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "rewritten prompt"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.minimax_api_call("system", "user", temperature=1.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 1.0)

    @patch("openai.OpenAI")
    def test_temperature_clamped_low(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "rewritten prompt"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.minimax_api_call("system", "user", temperature=-0.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)

    @patch("openai.OpenAI")
    def test_temperature_zero_accepted(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.minimax_api_call("system", "user", temperature=0.0, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.0)

    @patch("openai.OpenAI")
    def test_temperature_normal_passthrough(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.minimax_api_call("system", "user", temperature=0.7, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)


class TestMiniMaxClientThinkTagStrip(unittest.TestCase):
    """Test <think>...</think> parsing in MiniMaxClient."""

    @patch("openai.OpenAI")
    def test_think_tag_stripped(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>I need to rewrite this prompt</think>A beautiful sunset scene"
        )
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        thinking, result = client.minimax_api_call("system", "user", 0.1, 4096)

        self.assertEqual(thinking, "I need to rewrite this prompt")
        self.assertEqual(result, "A beautiful sunset scene")

    @patch("openai.OpenAI")
    def test_no_think_tag(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A beautiful sunset scene"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        thinking, result = client.minimax_api_call("system", "user", 0.1, 4096)

        self.assertEqual(thinking, "")
        self.assertEqual(result, "A beautiful sunset scene")

    @patch("openai.OpenAI")
    def test_empty_content(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        thinking, result = client.minimax_api_call("system", "user", 0.1, 4096)

        self.assertEqual(thinking, "")
        self.assertEqual(result, "")


class TestMiniMaxClientRunSingleRecaption(unittest.TestCase):
    """Test run_single_recaption method."""

    @patch("openai.OpenAI")
    def test_returns_result_only(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>reasoning</think>A cinematic scene of a cat"
        )
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        result = client.run_single_recaption("system_prompt", "A cat video")

        self.assertEqual(result, "A cinematic scene of a cat")

    @patch("openai.OpenAI")
    def test_default_temperature(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.run_single_recaption("system", "user")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.1)
        self.assertEqual(call_kwargs["max_tokens"], 4096)


class TestMiniMaxClientAPICall(unittest.TestCase):
    """Test API call parameters."""

    @patch("openai.OpenAI")
    def test_correct_api_params(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test-key", model_name="MiniMax-M2.5-highspeed")
        client.minimax_api_call("sys prompt", "user prompt", 0.3, 2048)

        mock_openai_cls.assert_called_once_with(
            base_url="https://api.minimax.io/v1",
            api_key="test-key",
            timeout=600,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5-highspeed")
        self.assertEqual(call_kwargs["temperature"], 0.3)
        self.assertEqual(call_kwargs["max_tokens"], 2048)
        self.assertFalse(call_kwargs["stream"])

    @patch("openai.OpenAI")
    def test_messages_format(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        client = MiniMaxClient(api_key="test")
        client.minimax_api_call("my system", "my user input", 0.1, 4096)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "my system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "my user input")

    @patch("openai.OpenAI")
    def test_retry_on_failure(self, mock_openai_cls):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "success"
        mock_client.chat.completions.create.side_effect = [
            Exception("rate limited"),
            mock_response,
        ]

        client = MiniMaxClient(api_key="test")
        with patch("time.sleep"):
            thinking, result = client.minimax_api_call("sys", "user", 0.1, 100)

        self.assertEqual(result, "success")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)


class TestRewriteUtilsProviderSelection(unittest.TestCase):
    """Test _create_t2v_client provider selection."""

    def test_minimax_provider_selected(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {
            "REWRITE_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key",
        }, clear=False):
            client = _create_t2v_client()
            self.assertIsInstance(client, MiniMaxClient)
            self.assertEqual(client.api_key, "test-key")
            self.assertEqual(client.model_name, "MiniMax-M2.7")

    def test_minimax_provider_custom_model(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {
            "REWRITE_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key",
            "T2V_REWRITE_MODEL_NAME": "MiniMax-M2.5-highspeed",
        }, clear=False):
            client = _create_t2v_client()
            self.assertIsInstance(client, MiniMaxClient)
            self.assertEqual(client.model_name, "MiniMax-M2.5-highspeed")

    def test_minimax_provider_case_insensitive(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        with patch.dict(os.environ, {
            "REWRITE_PROVIDER": "MiniMax",
            "MINIMAX_API_KEY": "test-key",
        }, clear=False):
            client = _create_t2v_client()
            self.assertIsInstance(client, MiniMaxClient)

    def test_default_provider_is_qwen(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client
        from hyvideo.utils.rewrite.clients import QwenClient

        with patch.dict(os.environ, {
            "T2V_REWRITE_BASE_URL": "http://localhost:8000/v1",
            "T2V_REWRITE_MODEL_NAME": "Qwen3-235B",
        }, clear=False):
            # Remove REWRITE_PROVIDER if present
            os.environ.pop("REWRITE_PROVIDER", None)
            client = _create_t2v_client()
            self.assertIsInstance(client, QwenClient)

    def test_default_provider_raises_without_env(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                _create_t2v_client()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_empty_provider_falls_back_to_qwen(self):
        from hyvideo.utils.rewrite.rewrite_utils import _create_t2v_client
        from hyvideo.utils.rewrite.clients import QwenClient

        with patch.dict(os.environ, {
            "REWRITE_PROVIDER": "",
            "T2V_REWRITE_BASE_URL": "http://localhost:8000/v1",
            "T2V_REWRITE_MODEL_NAME": "Qwen3-235B",
        }, clear=False):
            client = _create_t2v_client()
            self.assertIsInstance(client, QwenClient)


class TestT2VRewriteWithMiniMax(unittest.TestCase):
    """Test t2v_rewrite function with MiniMax provider."""

    @patch("openai.OpenAI")
    def test_t2v_rewrite_minimax(self, mock_openai_cls):
        from hyvideo.utils.rewrite.rewrite_utils import t2v_rewrite

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A cinematic wide shot of a girl"
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {
            "REWRITE_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key",
        }, clear=False):
            result = t2v_rewrite("A girl")

        self.assertEqual(result, "A cinematic wide shot of a girl")

    def test_t2v_rewrite_with_explicit_client(self):
        """Passing explicit rewrite_client should bypass provider selection."""
        from hyvideo.utils.rewrite.rewrite_utils import t2v_rewrite

        mock_client = MagicMock()
        mock_client.run_single_recaption.return_value = "rewritten"

        result = t2v_rewrite("test prompt", rewrite_client=mock_client)
        self.assertEqual(result, "rewritten")
        mock_client.run_single_recaption.assert_called_once()


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for MiniMax provider (require MINIMAX_API_KEY)."""

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    def test_minimax_real_api_call(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        client = MiniMaxClient()
        result = client.run_single_recaption(
            "You are a helpful assistant. Reply briefly.",
            "Say hello in one sentence.",
            temperature=0.1,
            max_tokens=50,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    def test_minimax_t2v_rewrite_integration(self):
        from hyvideo.utils.rewrite.rewrite_utils import t2v_rewrite

        with patch.dict(os.environ, {"REWRITE_PROVIDER": "minimax"}, clear=False):
            result = t2v_rewrite("A cat sitting on a table")

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    def test_minimax_m25_highspeed_model(self):
        from hyvideo.utils.rewrite.clients import MiniMaxClient

        client = MiniMaxClient(model_name="MiniMax-M2.5-highspeed")
        result = client.run_single_recaption(
            "You are a helpful assistant. Reply briefly.",
            "Say hello in one sentence.",
            temperature=0.1,
            max_tokens=50,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
