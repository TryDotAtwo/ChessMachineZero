# VM inspector site

The static site presents the exact currently executable artifact boundary. It
replays a fixed browser fixture for interaction; it does not pretend to execute
the native CUDA artifact in JavaScript. The native evidence and tensor shapes
shown in the UI are pinned by `tests/test_site_contract.py`.

Serve locally from the repository root:

```powershell
python -m http.server 4173 --directory site
```

GitHub Pages publishes only this directory through `.github/workflows/pages.yml`
after the change reaches `main` and Pages is configured to use GitHub Actions.
