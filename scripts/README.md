# Scripts

Operational entrypoints belong here.

Keep them thin:

- parse arguments
- load configs
- call reusable code from `src/phantom/`
- emit manifests and metrics

Do not hide core data or training logic in scripts.
