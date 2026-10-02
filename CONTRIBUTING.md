# Contributing to Claveles

Thank you for your interest in contributing to `Claveles`! This document provides guidelines and instructions for contributing to this repository.

---

## Developer Certificate of Origin (DCO)

To ensure that all contributions are legally compliant with open-source standards, this project requires contributors to certify their commits under the **Developer Certificate of Origin (DCO)**.

By contributing to this project, you agree to the terms outlined in the [DCO](./DCO) file.

### How to Sign Off Your Commits

To indicate your acceptance of the DCO, you must include a `Signed-off-by` line in every git commit message:

---

Signed-off-by: Your Name <your.email@example.com>

---

The easiest way to do this is to use the `-s` (or `--signoff`) flag when committing:

```sh
git commit -s -m "feat: add new feature to core"
```

> **Note:** Pull requests containing commits without a valid `Signed-off-by` line will not pass CI checks and cannot be merged.

---

## Repository Structure

This repository contains two Python package configurations:

* `./` (Root): Contains `Claveles-core` (the main open-source Python logic).
* `./meta`: Contains `Claveles` (the meta-package specifying dependencies).

---

## Getting Started

### 1. Prerequisites

Please see [README](README.md). You will also need Git to contribute.

### 2. Setting Up Your Development Environment

Fork and clone the repository, then set up an editable installation for development:

```sh
# Clone your fork
git clone https://github.com/your-username/claveles.git
cd claveles

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install build dependencies and development package in editable mode
pip install --upgrade pip
pip install -e .
```

---

## Development & Testing Workflow

### 1. Create a Branch

Create a feature or bugfix branch off of the `main` branch:

```sh
git checkout -b feature/your-feature-name
```

### 2. License Headers

If you create **new source files** (e.g., `.py`, `.proto`), please ensure they include the project's standard license header. You can automatically format and check headers using `make`:

```sh
# Automatically add missing license headers
make license-add

# Verify license headers locally before committing
make license-check
```

### 3. Make Changes & Commit

Write your code, update tests, apply license headers, and make sure to sign off your commits:

```sh
git add .
git commit -s -m "fix: resolve issue with core processing"
```

### 4. Verify Build locally

If you modify configuration files or metadata, test the build locally before submitting:

```sh
# Test building Claveles-core
python -m build

# Test building Claveles (meta-package)
python -m build meta/ --sdist
```

---

## Submitting a Pull Request (PR)

1. Push your branch to your GitHub fork:

```sh
git push origin feature/your-feature-name
```

2. Open a Pull Request against the `main` branch of this repository.
3. Ensure all CI status checks (including DCO verification and license header checks) pass.
4. Maintainers will review your PR and provide feedback.

Thank you for helping improve `Claveles`!
