"""Unit tests for the provider timeout status message.

When a provider request runs out of time and the client's retries are gone, the
provider used to return an empty string. The loop then had nothing to run, so
the turn ended without a word to the user and the task looked abandoned (#321).
The timeout now comes back as a `send` command carrying a status message, the
same way a reply cut off by the token limit already does.

No container, no network, no API key, and no provider SDK: `openai` and the
configuration module are stubbed before the module under test is loaded, the
same pattern as test_openclaw_unit.py.
"""
import importlib.util
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_LLM_EXT_PATH = os.path.join(_REPO_ROOT, "providers", "lib_llm_ext.py")
_HELPER_PATH = os.path.join(_REPO_ROOT, "src", "helper.py")

# lib_llm_ext.py does `from src.helper import quote_arg` (repo-root package).
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _StubAPITimeoutError(Exception):
    """Stands in for openai.APITimeoutError: the client's own timeout."""


def _install_stubs():
    config_stub = types.ModuleType("config")
    config_stub.config_get_by_key = lambda key, default=None: None
    sys.modules["config"] = config_stub

    openai_stub = types.ModuleType("openai")
    openai_stub.APITimeoutError = _StubAPITimeoutError
    openai_stub.OpenAI = object  # only referenced in a type annotation
    sys.modules["openai"] = openai_stub


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def llm():
    saved = {name: sys.modules.get(name) for name in ("config", "openai")}
    _install_stubs()
    try:
        yield _load("lib_llm_ext_under_test", _LIB_LLM_EXT_PATH)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def helper():
    return _load("helper_under_test", _HELPER_PATH)


def _gateway_error(status):
    error = Exception(f"{status} Gateway Time-out")
    error.status_code = status
    return error


class _RaisingClient:
    """Stand-in for the OpenAI client whose chat call always fails."""

    def __init__(self, error):
        def create(**kwargs):
            raise error

        completions = types.SimpleNamespace(create=create)
        self.chat = types.SimpleNamespace(completions=completions)


def _provider(llm, error):
    provider = llm.AIProvider("OpenAIAPI", "OPENAIAPI_API_KEY", "test-model", "http://localhost/v1/")
    provider._client = _RaisingClient(error)
    return provider


# --- which failures count as a timeout ---------------------------------------

def test_client_timeout_is_a_timeout(llm):
    assert llm._is_timeout_error(_StubAPITimeoutError("timed out"))


@pytest.mark.parametrize("status", [408, 504, 524])
def test_gateway_timeout_statuses_are_a_timeout(llm, status):
    assert llm._is_timeout_error(_gateway_error(status))


@pytest.mark.parametrize("status", [400, 429, 500, 502])
def test_other_statuses_are_not_a_timeout(llm, status):
    assert not llm._is_timeout_error(_gateway_error(status))


def test_a_plain_error_is_not_a_timeout(llm):
    assert not llm._is_timeout_error(ValueError("boom"))


# --- what chat() returns ------------------------------------------------------

def test_chat_tells_the_user_when_the_client_times_out(llm):
    result = _provider(llm, _StubAPITimeoutError("timed out")).chat("prompt")
    assert result == llm._llm_timeout_command()
    assert "timed out" in result


def test_chat_tells_the_user_when_the_gateway_times_out(llm):
    assert _provider(llm, _gateway_error(504)).chat("prompt") == llm._llm_timeout_command()


def test_chat_still_returns_nothing_for_other_failures(llm):
    assert _provider(llm, ValueError("boom")).chat("prompt") == ""


# --- the message has to survive the parser the loop runs it through ----------

def test_the_timeout_command_parses_into_one_send(llm, helper):
    parsed = helper.balance_parentheses(llm._llm_timeout_command())
    assert parsed.startswith('((send "')
    assert parsed.endswith('"))')
    assert parsed.count("(send ") == 1


# --- one attempt, so the timeout is reported when the first request gives up --

def _client_kwargs(llm, monkeypatch, gateway):
    captured = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm.openai, "OpenAI", recorder)
    monkeypatch.setattr(
        llm, "config_get_by_key",
        lambda key, default=None: gateway if key == "GATEWAY_URL" else default,
    )
    if gateway is None:
        monkeypatch.setenv("OPENAIAPI_API_KEY", "dummy")
    provider = llm.AIProvider("OpenAIAPI", "OPENAIAPI_API_KEY", "test-model", "http://localhost/v1/")
    assert provider._create_client() is not None
    return captured


def test_the_proxy_client_makes_one_attempt(llm, monkeypatch):
    assert _client_kwargs(llm, monkeypatch, "http://localhost:8080")["max_retries"] == 0


def test_the_direct_client_makes_one_attempt(llm, monkeypatch):
    assert _client_kwargs(llm, monkeypatch, None)["max_retries"] == 0
