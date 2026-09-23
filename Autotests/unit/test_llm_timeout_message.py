"""Unit tests for the provider timeout status message.

When a provider request runs out of time and the client's retries are gone, the
provider used to return an empty string. The loop then had nothing to run, so
the turn ended without a word to the user and the task looked abandoned (#321).
The timeout now comes back as a `send` command carrying a status message, the
same way a reply cut off by the token limit already does.

No container, no network, no API key: the client is replaced by a stub that
raises the error under test.
"""
import importlib.util
import os
import sys

import httpx
import openai
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO_ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def llm():
    return _load("lib_llm_ext_under_test", os.path.join("providers", "lib_llm_ext.py"))


@pytest.fixture(scope="module")
def helper():
    return _load("helper_under_test", os.path.join("src", "helper.py"))


class _RaisingClient:
    """Minimal stand-in for openai.OpenAI whose chat call always fails."""

    def __init__(self, error):
        completions = type("_Completions", (), {"create": lambda _self, **kwargs: (_ for _ in ()).throw(error)})()
        self.chat = type("_Chat", (), {"completions": completions})()


def _provider(llm, error):
    provider = llm.AIProvider("OpenAIAPI", "OPENAIAPI_API_KEY", "test-model", "http://localhost/v1/")
    provider._client = _RaisingClient(error)
    return provider


def _timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "http://gateway/openaiapi/chat/completions"))


# --- which failures count as a timeout ---------------------------------------

def test_client_timeout_is_a_timeout(llm):
    assert llm._is_timeout_error(_timeout_error())


@pytest.mark.parametrize("status", [408, 504, 524])
def test_gateway_timeout_statuses_are_a_timeout(llm, status):
    error = Exception("gateway timeout")
    error.status_code = status
    assert llm._is_timeout_error(error)


@pytest.mark.parametrize("status", [400, 429, 500, 502])
def test_other_statuses_are_not_a_timeout(llm, status):
    error = Exception("other failure")
    error.status_code = status
    assert not llm._is_timeout_error(error)


def test_a_plain_error_is_not_a_timeout(llm):
    assert not llm._is_timeout_error(ValueError("boom"))


# --- what chat() returns ------------------------------------------------------

def test_chat_tells_the_user_when_the_request_times_out(llm):
    result = _provider(llm, _timeout_error()).chat("prompt")
    assert result == llm._llm_timeout_command()
    assert "timed out" in result


def test_chat_tells_the_user_when_the_gateway_times_out(llm):
    error = openai.InternalServerError(
        "504 Gateway Time-out",
        response=httpx.Response(504, request=httpx.Request("POST", "http://gateway/openaiapi/chat/completions")),
        body=None,
    )
    assert _provider(llm, error).chat("prompt") == llm._llm_timeout_command()


def test_chat_still_returns_nothing_for_other_failures(llm):
    assert _provider(llm, ValueError("boom")).chat("prompt") == ""


# --- the message has to survive the parser the loop runs it through ----------

def test_the_timeout_command_parses_into_one_send(llm, helper):
    parsed = helper.balance_parentheses(llm._llm_timeout_command())
    assert parsed.startswith('((send "')
    assert parsed.endswith('"))')
    assert parsed.count("(send ") == 1
