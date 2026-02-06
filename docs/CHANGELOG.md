# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI package distribution (`uv pip install dhbv2`)
- Automated wheel building and publishing via GitHub Actions

### Changed
- Switched build system from setuptools to hatchling + hatch-vcs
- Moved dev tools (`pre-commit`, `uv`) from runtime to dev dependencies

## [0.1.0] - 2025-XX-XX

### Added
- Daily BMI adapter (`DeltaModelBmi`) for δHBV 2.0
- Hourly Multi-TimeScale BMI adapter (`MtsDeltaModelBmi`) for δHBV 2.0 MTS
- PET calculation utilities (Hargreaves, Penman-Monteith)
- RingBuffer utility for rolling window input caching
- NextGen/ngen integration with BMI interface
- Docker support for NGIAB deployment
- Documentation for setup, data, standalone/NextGen/NGIAB execution, routing, and validation
