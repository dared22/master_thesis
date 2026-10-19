# Run `make thesis` at the repository root.
# From any repository directory:
# make -C "$(git rev-parse --show-toplevel)" thesis
.DEFAULT_GOAL := thesis
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: thesis
thesis:
	@"$(REPO_ROOT)thesis/build.sh"
