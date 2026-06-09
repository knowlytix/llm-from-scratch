# Contributing to lm-from-scratch

Thanks for your interest in contributing! This repository is the **open**
(lm-from-scratch) part of the project — it runs with no license. The advanced GMS
features live in the separately-licensed `knowlytix` backend and are **not**
developed here; please keep contributions to the open baseline.

## Development setup

```bash
git clone https://github.com/knowlytix/llm-from-scratch.git
cd llm-from-scratch
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch
pip install -e ".[dev]"
pytest -q
```

Python 3.12+ is required. The GMS-gated tests skip automatically when `knowlytix`
is not installed — that is expected; CI runs the same way.

## Making a change

1. Open an issue first for anything non-trivial, so we can agree on the approach.
2. Keep PRs focused — one logical change per PR.
3. Add or update tests; `pytest -q` must pass.
4. Match the surrounding code style (the repo uses `ruff` and `black`; run them
   if you have the `[dev]` extra installed).
5. Update docs/README if behavior changes.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/)
instead of a CLA. It is a lightweight statement that you wrote the patch or
otherwise have the right to submit it under the project's Apache-2.0 license.

**Sign off every commit** by adding a `Signed-off-by` line — `git commit -s` does
this for you using your `git` name and email:

```
Signed-off-by: Your Name <you@example.com>
```

PRs whose commits are not signed off will be asked to amend before merge
(`git commit --amend -s`, or `git rebase --signoff` for multiple commits).

By signing off you certify the DCO (full text at the link above): in short, that
you have the right to contribute the code and agree it is licensed under this
project's license.

## License

By contributing, you agree that your contributions are licensed under the
repository's [Apache-2.0](LICENSE) license (inbound = outbound).

Questions: **hello@knowlytix.ai**
