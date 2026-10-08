# Contributing

Thank you for helping make transformers easier to understand. This guide covers how changes are proposed and the standards for content.

## Ways to contribute

- **Report errors.** Mistakes in math, code or explanations matter most. Open an issue with the section, what is wrong and, if possible, a source.
- **Flag confusing parts.** Say which step lost you. Feedback from newcomers is especially valuable.
- **Improve content.** Clearer wording, better examples, new exercises or visuals.
- **Improve code.** Bug fixes, performance, accessibility and tests.

## Development setup

You need Node.js 24 LTS (see `.nvmrc`) and npm.

```sh
npm install
npm run check
```

`npm run check` runs the format check, lint, type check and tests. `npm run format` fixes formatting.

## Workflow

1. Open or pick an issue so work is not duplicated.
2. Branch from `main` using a type prefix, for example `feat/attention-view` or `fix/softmax-rounding`.
3. Keep each pull request to one feature or fix. Split large features into several pull requests that each leave `main` working.
4. Title pull requests with [Conventional Commits](https://www.conventionalcommits.org/), for example `feat: add attention heatmap`.
5. Write commit messages and pull request descriptions for other engineers. Keep them objective and concise.
6. Complete the pull request template, including screenshots for visual changes.
7. Add a label so the release notes can sort it, such as `enhancement`, `bug`, `content`, `accessibility` or `documentation`.
8. Pull requests are rebase-merged after checks pass and a maintainer approves, so each commit stays in the history. Keep each commit a working step with a clear message.

## Releases

Versions follow [Semantic Versioning](https://semver.org/). Each feature that readers notice, such as a new page or chapter, gets a minor version, and fixes get a patch version. A change that breaks how the engine or the model files are used gets a major version.

Each release has a milestone that collects its issues. To make a release, a maintainer:

1. Updates `version` and `date-released` in `CITATION.cff`, in the last pull request of the release. The packages are private, so their versions stay at 0.0.0 and the git tag is the version.
2. After it merges, creates the release from `main` with `gh release create v<version> --generate-notes`. The notes are grouped by the labels above.
3. Closes the milestone.

## Content standards

### Writing

- Use short sentences and plain words. Define a term before using it.
- Avoid em dashes and colons in running text.
- Prefer precision to vagueness. When simplifying, say so.
- Stay faithful to the sources you cite.
- When you use an analogy, say where it breaks.

### Citations

- Cite a primary source for every non-obvious claim, preferably the original paper.
- Give the exact location, such as "Vaswani et al. 2017, §3.2.1, Eq. 1". If you do not know it yet, you can add it in a later pull request.
- Link to a pinned version of the paper, such as `https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1`, so numbering cannot drift.
- Label each citation as **From the paper**, **Later research** or **Our simplification**.
- Give each citation a short, exact quote from the source.

### Figures and third-party material

- Create original figures. Do not copy figures from other works unless their license allows it.
- List any third-party material and its license in the pull request.

## Licensing of contributions

By contributing, you agree that code is released under the [MIT License](LICENSE) and content under [CC BY 4.0](LICENSE-CONTENT). Code samples inside the lessons are also available under the MIT License.
