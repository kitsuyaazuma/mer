GO ?= go
GOLANGCI_LINT ?= golangci-lint

.PHONY: build vet fmt lint test tidy check data

build:
	$(GO) build ./...

vet:
	$(GO) vet ./...

fmt:
	$(GO) fmt ./...

lint:
	$(GOLANGCI_LINT) run ./...

test:
	$(GO) test ./...

tidy:
	$(GO) mod tidy

check: fmt tidy build vet lint test

data:
	./scripts/download-data.sh
	$(GO) run ./cmd/prepare
