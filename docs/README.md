# Physics-R1 — Project Page

This directory is the source of the project page served at
**<https://shanyang-me.github.io/physics-r1-neurips2026/>**.

## Files

- `index.html` — the page itself (single-file site).
- `static/css/`, `static/js/` — Bulma + Bulma extensions; FontAwesome.
- `static/pdfs/` — paper PDF lives here as `physics-r1.pdf`.
- `static/images/` — figures, social-preview, favicon.

## Local preview

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000/
```

## Deploy

GitHub Pages → repository **Settings → Pages → Build and deployment**:
- Source: **Deploy from a branch**
- Branch: `main` / folder: `/docs`

## Updating the paper PDF

The paper PDF is referenced at `static/pdfs/physics-r1.pdf`. To update:

```bash
cp /path/to/main.pdf docs/static/pdfs/physics-r1.pdf
git add docs/static/pdfs/physics-r1.pdf
git commit -m "docs: update paper PDF"
git push
```

## Credits

Built on top of the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
by Eliahu Horwitz.
