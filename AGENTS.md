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

Before making changes, inspect the relevant code and follow the
existing project structure and configuration.

Use the project's uv environment. After Python changes, run:
- uv run ruff check .
- uv run ruff format --check .
- uv run pyright
- Relevant tests for the changed behavior

These commands must use the repository's checked-in configuration.
If the repository documents different canonical commands, use those.

Resolve Ruff and Pyright errors in files you change, including
pre-existing errors. Re-run checks after fixes until they pass.

Do not suppress diagnostics, weaken checking rules, exclude files,
or replace useful types with Any merely to make checks pass.
Use suppressions only when technically justified, and explain them.

For repository-wide cleanup tasks, resolve all diagnostics within
the configured checking scope, including pre-existing diagnostics.
Continue fixing and re-running checks until they pass or a concrete
blocker requires user input.

At completion, report the commands run and their results. Identify
remaining failures and blockers explicitly. Do not describe work
as fully validated when required checks are failing.