import pytest
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

BASE = {"model": "z-ai/glm-5.3-flash", "messages": [{"role": "user", "content": "OK"}]}


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, {"thinking_budget": 8192}),
        ({"thinking_token_budget": 0}, {"thinking_budget": 0}),
        ({"thinking_token_budget": 4096}, {"thinking_budget": 4096}),
        ({"thinking_token_budget": None}, None),
        ({"custom_params": {"other": 1}}, {"other": 1, "thinking_budget": 8192}),
        (
            {"thinking_token_budget": 4096, "custom_params": {"thinking_budget": 2048}},
            {"thinking_budget": 2048},
        ),
    ],
)
def test_budget_contract(extra, expected):
    request = ChatCompletionRequest.model_validate({**BASE, **extra})
    assert request.custom_params == expected


@pytest.mark.parametrize("budget", [True, -1, "8192", 1.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({**BASE, "thinking_token_budget": budget})
