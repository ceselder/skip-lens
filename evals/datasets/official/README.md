# Official Global Workspace prompt sets

These files are copied without semantic modification from
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), the
companion repository for *Verbalizable Representations Form a Global Workspace
in Language Models* (Gurnee et al., 2026).

Upstream commit used by this experiment:
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The upstream project and its synthetic prompt data are Apache License 2.0; its
license is reproduced in this directory. All six method-comparison sets and
all eleven released appendix experiment configurations are included. See the
copied `UPSTREAM_README.md` files for the exact task conventions. In particular,
poetry is read at the last newline, not at the final prompt token; the other
evaluation-specific positions follow the upstream README.

The older JSON files one directory above are locally regenerated substitutes.
They remain for reproducibility of earlier runs but are not used by the new
official evaluation harness.
