# Security

transformer-visualized is a static site. It has no server, no accounts and no cookies, and it sends nothing you type anywhere. The model runs in your browser.

## Reporting a problem

Please report a security problem privately, with **Report a vulnerability** on the repository's Security tab. Please don't open a public issue for it.

Say what you found, how to see it, and which browser you used. You'll get a reply within a week.

## What is in place

- Every page carries a content security policy. Scripts and styles must come from the site itself, and each inline one is allowed by its hash.
- Everything the site loads comes from its own address. Links to papers and other sites only open them.
- Workflow actions are pinned to exact commits, and Dependabot keeps them and the npm packages up to date.
