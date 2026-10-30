# BitGenesis Security

BitGenesis is an early-stage experimental project. It is not designed to
protect data or systems in high-risk environments and makes no security or
privacy guarantees beyond those provided by the environment in which it runs.

## Reporting a vulnerability

If you discover a vulnerability, do not open a public issue or pull request
with details that could enable its exploitation.

Report it through GitHub's private **Report a vulnerability** feature, if it
is available for this repository. If the feature is unavailable, contact the
maintainers privately through a verifiable contact channel associated with
the repository or their GitHub profiles. Do not publish details before they
have had an opportunity to respond.

If possible, include:

- a clear description of the impact and required conditions;
- the minimum steps needed to reproduce the issue, using only systems and
  data you are authorized to test;
- the affected version or commit, operating system, and relevant configuration;
- any known mitigations.

Do not include secrets, credentials, personal information, private corpora, or
checkpoints in a report. If they are essential to demonstrate the issue,
describe how to reproduce it with synthetic data instead.

The maintainers will try to acknowledge and assess reports within a reasonable
time, but no response or remediation timeline is currently defined. Coordinate
public disclosure with the maintainers after a fix or mitigation is available.
There is no published support matrix; support for previous versions is not
guaranteed.

## Data and local use

Datasets, `Genesis.md` files, checkpoints, and local reports may contain
sensitive information. Keep them out of public repositories and do not share
them in reports. BitGenesis is intended to run locally, but the security of
files and the host also depends on the user's permissions, configuration, and
installed tools.

Generated responses are experimental and may be incorrect or unsafe. Review
them before running code, applying changes, or making decisions.
