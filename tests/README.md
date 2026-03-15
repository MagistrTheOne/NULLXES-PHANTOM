# Tests

The first test layer should be:

- smoke tests for imports, configs, and tiny forwards
- integration tests for data -> tokenize -> pack -> train-step -> eval flows

Large distributed tests should remain opt-in and separate from the default fast suite.
