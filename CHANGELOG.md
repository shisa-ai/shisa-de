# Changelog

## 0.1.2

- Serialize lazy tokenizer imports and loads across client instances to prevent
  concurrent first-call initialization races.
- Keep initialized tokenizer access and normal requests outside the lock, and
  allow initialization to be retried after a failure.

## 0.1.1

- Use `pip install shisa-de` in the README.
- Add this changelog.
- No API or scoring changes.

## 0.1.0

- Initial PyPI release.
- Classify text and images through Shisa Platform or a local DE-1 server.
- Ask typed questions with `Noul`, `Choice`, and `Score`.
- Add `doctor`, `ask`, and `explain` CLI commands.
