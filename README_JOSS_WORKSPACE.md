# JOSS submission workspace

Folder for the [*Journal of Open Source Software*](https://joss.theoj.org/) track (renamed from `ComputationalStatistics` / SoftwareX archive).

| Path | Role |
|------|------|
| `src/localmapbench/` | Installable Python package (skeleton → full API) |
| `software/` | Legacy research scripts & frozen tables (migration source) |
| `manuscript/` | Prior LSSP LaTeX (reference only; **not** the JOSS paper) |
| `docs/` | Suitability, TODOs, venue notes |
| `tests/` | Pytest suite |
| `paper.md` | JOSS paper (to add later) |

## Work 1 status

- [x] Folder named for JOSS
- [x] MIT `LICENSE`
- [x] Package skeleton + README + CONTRIBUTING + CITATION
- [ ] **Public GitHub repository** (create + push — see below)

## Create the public GitHub repo (required to start the 6-month clock)

If `gh` is available and authenticated:

```powershell
cd JOSS
git init -b main
# after local commits:
gh repo create localmapbench --public --source=. --remote=origin --push
```

Or create an empty public repo named `localmapbench` on GitHub, then:

```powershell
git remote add origin https://github.com/<YOUR_USER>/localmapbench.git
git push -u origin main
```

Update `pyproject.toml` `[project.urls]` if your GitHub username differs from the placeholder.
