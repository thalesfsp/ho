###
# Params.
###

PROJECT_NAME := "ho"
BIN_NAME := $(PROJECT_NAME)
BIN_DIR := bin
BIN_PATH := $(BIN_DIR)/$(BIN_NAME)

HAS_GODOC := $(shell command -v godoc;)
HAS_GOLANGCI := $(shell command -v golangci-lint;)
HAS_GORELEASER := $(shell command -v goreleaser;)

default: ci

###
# Entries.
###

build:
	@go build -o $(BIN_PATH) && echo "Build OK"

build-dev:
	@go build -gcflags="all=-N -l" -o $(BIN_PATH) && echo "Build OK"

ci: build lint

doc:
ifndef HAS_GODOC
	@echo "Could not find godoc, installing it"
	@go install golang.org/x/tools/cmd/godoc@latest
endif
	@echo "Open localhost:6060/pkg/github.com/thalesfsp/$(PROJECT_FULL_NAME)/ in your browser\n"
	@godoc -http :6060

lint:
ifndef HAS_GOLANGCI
	@echo "Could not find golangci-list, installing it"
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.61.0
endif
	@golangci-lint run -v -c .golangci.yml && echo "Lint OK"

release-local:
ifndef HAS_GORELEASER
	@echo "Could not find goreleaser, installing it"
	@go install github.com/goreleaser/goreleaser/v2@v2.3.2
endif
	@goreleaser build --clean --snapshot && echo "Local release OK"

.PHONY: build \
	build-dev \
	ci \
	doc \
	lint \
	release-local
