# Publish to PyPI

Releases use GitHub Actions and PyPI Trusted Publishing. No PyPI API token is
stored in GitHub.

## One-time setup

Register a pending publisher at <https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| Project | `shisa-de` |
| GitHub owner | `shisa-ai` |
| Repository | `shisa-de` |
| Workflow | `publish.yml` |
| Environment | `pypi-publish` |

The GitHub environment `pypi-publish` allows `v*` tags and requires approval
from `lhl`. Self-approval is enabled so the release author can approve the job.
The first successful upload creates the PyPI project.

## Release checklist

1. Check `git status -sb` and keep unrelated work out of the release.
2. Update both version fields: `pyproject.toml` and `shisa_de/__init__.py`.
3. Run `python -m pytest tests/` and review the README examples.
4. Commit the release files, then push the commit and an annotated tag:

   ```bash
   git push origin main
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin v0.1.0
   ```

   Replace `0.1.0` with the version being released.

5. Open [Publish to PyPI](https://github.com/shisa-ai/shisa-de/actions/workflows/publish.yml).
   The build checks version consistency, runs tests, builds and checks both
   distributions, and smoke-tests the installed wheel and tokenizer.
6. Once the build passes, review the run and approve the `pypi-publish` deployment.
7. Verify the published install:

   ```bash
   uvx --refresh --from 'shisa-de==0.1.0' shisa-de --help
   ```

8. Check <https://pypi.org/project/shisa-de/>. After the first release, replace
   the README's GitHub install command with `python -m pip install shisa-de`.

PyPI does not allow replacing uploaded distribution files. Release a new
version for fixes; do not move a published version tag.
