# Contributing to tauri-plugin-ai

Thanks for your interest in contributing! Here's how to get involved.

## Reporting Bugs

Use the [bug report issue template](https://github.com/user/tauri-plugin-ai/issues/new?template=bug_report.md). Include which AI provider and feature flags you're using.

## Suggesting Features

Use the [feature request issue template](https://github.com/user/tauri-plugin-ai/issues/new?template=feature_request.md). Describe the use case and any alternatives you've considered.

## Submitting Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Open a PR against `main`

## Code Style

- **Rust only** -- this is a Tauri plugin with no frontend framework code.
- Run `cargo fmt` before committing.
- Zero warnings on `cargo build`.

## Testing

All of the following must pass before submitting a PR:

```bash
cargo build
cargo build --features local-onnx
npx tsc --noEmit
```

## Security

API keys must never reach the renderer process. All secret handling stays in the Rust backend. If your change touches credential flow, call this out explicitly in your PR description.
