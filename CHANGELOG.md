# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-08-25
### Added
- MIT license and this changelog.
- `requirements.txt` mirroring the `pyproject.toml` dependencies plus the
  notebook tooling used in `offline/`.

### Notes
- The IDS peak SDK wheels (`ids-peak`, `ids-peak-ipl`) are installed separately
  with `--no-deps` after the IDS peak Cockpit.
- `legacy/` holds the pre-rewrite sources for reference only and is not
  maintained.
