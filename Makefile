.PHONY: test install-hooks

VENV := .venv/bin

test:
	@echo "Running end-to-end tests (service must be on localhost:7860)..."
	$(VENV)/pytest tests/test_e2e.py -v

install-hooks:
	@cp scripts/pre-push .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "pre-push hook installed — tests will run before every push"
