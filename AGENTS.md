## Commit messages

Use Conventional Commits: `<type>(optional scope): <description>`.

Use types such as `feat`, `fix`, `docs`, `refactor`, `test`, and `chore`. Keep the subject concise and focused on the main change.

For a small, self-contained change, use a one-line commit message. For a change with several meaningful parts, add a body after a blank line. Use bullets to summarize the main changes, and include context or reasoning when it helps explain why the change was made.

When asked to commit, inspect the staged diff and write a message that accurately describes it. Don’t claim changes that aren’t in the diff.

Examples:

```text
docs: add setup instructions for the local environment
```

```text
feat(notebooks): add project case comparison

- Add a notebook to compare implementation evidence across cases
- Include source links and a summary of key findings
- Document assumptions that need further review
```

## Commit and push authorization

When I request a commit and push, proceed without additional confirmation to `https://github.com/sheisroot/digital-applications-prototype.git`.
The existing author email in project metadata is approved for publication there.
Use normal pushes; ask before force-pushing or deleting remote branches.

## Implementation and validation

Before making changes, inspect the relevant code and follow the existing project structure and configuration. Keep changes focused on the requested work.

For Python changes, run the configured Ruff lint and format checks, Pyright, and relevant tests. Use the project’s `uv` environment and the commands documented in `pyproject.toml` or the README.

Fix issues introduced by the change. If a check reports pre-existing issues outside the change, report them separately; don’t hide them with ignores or claim the check passed.

At the end, summarize the changes and list the validation commands run and their results. If a check could not be run, say why.