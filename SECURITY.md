# Security

## Threat model

docs-to-quiz is built for **self-hosting by a single trusted user**. There
is no authentication or multi-tenant isolation — anyone who can reach the
API can manage knowledge bases and generate questions.

**Do not expose this directly to the public internet.** Run it on
localhost, behind a VPN, or behind your own auth/reverse proxy if you need
remote access. This is a deliberate design decision, not an oversight —
see the README's [Design decisions](README.md#design-decisions) section.

## Reporting a vulnerability

For issues within the app's actual boundary (path traversal, injection,
anything that breaks the single-user threat model above rather than just
being "there's no login"), please report privately via [GitHub Security
Advisories](https://github.com/iqbal-rahmatullah/docs-to-quiz/security/advisories/new)
rather than a public issue.
