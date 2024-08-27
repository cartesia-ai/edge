# Contributing

Much of this guide was adopted from [AXLearn](https://github.com/apple/axlearn/blob/main/CONTRIBUTING.md).

## General Guidelines
1. Please do not commit broken code to any branch.
1. Only commit stable config files. Config files that are in development should not be committed.
1. Use pull requests (PR) to the merge in code. Do not develop on the main branch.

## Coding Style

We follow [the Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

To avoid confusion around naming, respect code consistency in the repo, including but not limited to naming conventions and code structure.

If you have not already, we recommend [setting up `pre-commit`](docs/01-start.md#optional-additional-setup-for-developers), which runs some of the linters/formatters prior to each commit. The same checks will be required to pass in CI, so this will help make the development process smoother.

### Type Annotations

Functions and methods must be annotated with types. We use [pytype](https://google.github.io/pytype/user_guide.html) for type checking.

## Code Review Process

### Author

All PRs must be made with the default template provided in the repository.

Before embarking on a major PR, send out a sketch PR (including the high level design notes in the PR description) to solicit feedback first.

It is more manageable for both the author and reviewers to have small PRs that can be quickly reviewed and merged.

When selecting multiple reviewers, use "Assignees" to indicate that approvals from specific
reviewers are required before merging.

The PR authors are expected to reply to each comment they have addressed, e.g., with "Done".
However, they should refrain from resolving the comments -- instead, let the reviewers do so.

When addressing a comment, pay attention to other places where the same comment may apply, so that
the reviewer does not have to repeat the comment.

When a PR is ready for another round of review, click [**re-request review**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) to notify the reviewer.

Although not strictly enforced, a general etiquette is to wait for all reviewers who have left comments to approve the PR prior to merging, even if the PR has already received an approval.

In some cases, the comments may be "nits" and appropriate for addressing in follow-up PRs. We encourage authors to give a heads up to reviewers prior to merging a PR with unresolved comments by leaving PR comments (e.g. "Will address in a follow-up") as well as [leaving a TODO](#leaving-todos) where applicable.

### Reviewer

People on the team should feel free to add themselves as reviewers and/or to assignees.

Consider prioritizing reviews over writing one's own code.
This will make the entire team more productive.

Code review does not end with merge of the PR.
Reviewers should feel free to add comments after the merge, which can be addressed in follow-up PRs.

## Attributions

Code that refers to (or is adapted from) other sources must explicitly reference the original source by providing a [link in the docstring](https://github.com/apple/axlearn/blob/669f0cae6249e165caa1a94cf64b12e77bf4cfdf/axlearn/common/attention.py#L360-L365) of the corresponding function, class, etc.

Code that is adapted from papers should have clear `Reference:` section in the docstring that provides a complete reference (with link) to the original paper. Links to ArXiv papers are preferred to avoid paywalls.

## Leaving TODOs

In some cases it's useful to track future work ("TODOs").
TODOs should have the format `TODO(username1,username2)` indicating the contributor(s) responsible for addressing the TODO.
Please use your actual GitHub username as opposed to an alias to avoid ambiguity.

For larger work items, consider creating a GitHub issue to track progress.
