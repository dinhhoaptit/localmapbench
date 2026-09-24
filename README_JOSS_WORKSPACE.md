# JOSS submission workspace

[*Journal of Open Source Software*](https://joss.theoj.org/) track.

Software name: **localmapbench**

## Done in work 1 (local)

- Renamed workspace folder to `JOSS/`
- MIT `LICENSE`
- Installable skeleton: `src/localmapbench/`, `pyproject.toml`, `tests/`
- README, CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff, CHANGELOG
- Local git with **4 iterative commits** (not a single dump)

## Remaining for work 1 (public)

Create and push a **public** GitHub repo so the JOSS 6-month clock starts:

```powershell
cd "C:\Users\hoand\OneDrive\Documents\space convolution\JOSS"
# Install GitHub CLI if needed, then:
gh auth login
gh repo create localmapbench --public --source=. --remote=origin --push
```

Or create `localmapbench` on github.com (public, empty), then:

```powershell
git remote add origin https://github.com/<YOUR_USER>/localmapbench.git
git push -u origin main
```

Update `pyproject.toml` URLs to match your GitHub username.

## Next (work 2+)

Migrate maps/protocol from `software/` into `src/localmapbench/`, add CI, then continue public commits over months. See `docs/JOSS_TODO.md`.
