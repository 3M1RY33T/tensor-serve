# Security Policy

## Supported Versions

Security fixes are provided for the latest released version of Tensor Serve.

## Reporting a Vulnerability

Please do not open a public GitHub issue for a suspected vulnerability. Use
GitHub's private vulnerability reporting feature for this repository, or contact
the maintainer through the repository profile.

Include:

- Affected version or commit
- Steps to reproduce
- Impact
- Any known workaround

Tensor Serve can store provider API keys in `config.json`. Secrets are encrypted
at rest, but deployments should provide `TENSOR_CONFIG_KEY` or
`TENSOR_CONFIG_KEY_FILE` from an OS keychain, secret manager, or protected
environment. Do not commit `.tensor_config.key`, `config.json`, or runtime state.
