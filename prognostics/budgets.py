"""Measured API usage and conservative admission limits for opt-in experiments."""

import json


class ExperimentBudget:
    def __init__(self, total_tokens=50000, max_requests=60):
        self.total_tokens = total_tokens
        self.max_requests = max_requests
        self.used = 0
        self.requests = 0
        self.unknown_usage = False

    def admit(self, messages, tools, limit):
        if self.unknown_usage or self.requests >= self.max_requests:
            raise ValueError("Experiment API budget exhausted or usage unknown.")
        # Conservative admission proxy, not a provider-specific tokenizer or monetary cap.
        prompt_bound = (
            len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode())
            + 256
        )
        limit = min(limit, self.total_tokens - self.used - prompt_bound)
        if limit < 512:
            raise ValueError("Insufficient remaining experiment budget for another request.")
        self.requests += 1
        return limit

    def record(self, response):
        if response.usage is None:
            self.unknown_usage = True
        else:
            self.used += response.usage.total_tokens
