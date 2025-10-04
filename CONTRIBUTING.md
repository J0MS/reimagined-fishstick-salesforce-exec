# Welcome to our contributing guidelines <!-- omit in toc -->

Thank you for investing your time in contributing to our project!

Read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

Use the table of contents icon <img src="https://user-images.githubusercontent.com/115469901/231885339-dc416e24-eabf-4904-9db0-223e664c748a.png" width="25" height="25" /> on the top left corner of this document to get to a specific section of this guide quickly.

## New contributor guide

To get an overview of the project, read the [README](README.md). Here are some resources to help you get started with contributions:

- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)


## Getting started

Generally, we expect you to clone the repository and work on a branch on your fork. This workflow allows you to make changes without affecting the main branch, and it gives you the flexibility to make changes without having to wait for a review. 

The following section is a general guideline for contributing to this repository.

### Issues

#### Create a new issue

If you spot a problem with the docs, [search if an issue already exists](https://github.com/ab-inbev-labs/ml-platform-observability/issues). If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/ab-inbev-analytics/ml-platform-observability/issues). 

#### Solve an issue

Scan through our [existing issues](https://github.com/ab-inbev-labs/ml-platform-observability/issues) to find one that interests you. You can narrow down the search using `labels` as filters. See [Labels](https://github.com/github/docs/blob/main/contributing/how-to-use-labels.md) for more information. As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

### Branching strategy

We are going to use [github-flow](https://docs.github.com/en/get-started/using-github/github-flow) branching strategy :rocket:


### Commit your update

Commit the changes once you are happy with them. Don't forget to [self-review](https://github.com/github/docs/blob/main/contributing/self-review.md) to speed up the review process:zap:.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Fill the "Ready for review" template so that we can review your PR. This template helps reviewers understand your changes as well as the purpose of your pull request. 
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a Docs team member will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations :tada::tada: The MLOps team thanks you :sparkles:. 


## Windows

This site can be developed on Windows, however a few potential gotchas need to be kept in mind:

1. Regular Expressions: Windows uses `\r\n` for line endings, while Unix-based systems use `\n`. Therefore, when working on Regular Expressions, use `\r?\n` instead of `\n` in order to support both environments. The Node.js [`os.EOL`](https://nodejs.org/api/os.html#os_os_eol) property can be used to get an OS-specific end-of-line marker.
2. Paths: Windows systems use `\` for the path separator, which would be returned by `path.join` and others. You could use `path.posix`, `path.posix.join` etc and the [slash](https://ghub.io/slash) module, if you need forward slashes - like for constructing URLs - or ensure your code works with either.
3. Bash: Not every Windows developer has a terminal that fully supports Bash, so it's generally preferred to write [scripts](/script) in JavaScript instead of Bash.
4. Filename too long error: There is a 260 character limit for a filename when Git is compiled with `msys`. While the suggestions below are not guaranteed to work and could cause other issues, a few workarounds include:
    - Update Git configuration: `git config --system core.longpaths true`
    - Consider using a different Git client on Windows

### Congrats!
You made it to the end of this guide! :tada: :tada: :tada:

Next, you can learn more about the commit message format, and how to write a good commit message. **MLOps standards** for pull requests, issues, and code reviews are also covered in the [Wiki](https://github.com/ab-inbev-analytics/ml-platform-observability/wiki).
