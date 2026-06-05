# PyPI Release — To Do

## 1. Create a PyPI account
- Go to https://pypi.org and sign up for a free account

## 2. Get an API token
- Log in → Account Settings → API tokens
- Create a token scoped to "Entire account"
- Copy and save it somewhere safe

## 3. Upload the package
Run in the terminal from the project root:

```cmd
python -m twine upload dist/*
```

When prompted:
- **Username:** `__token__`
- **Password:** paste your API token

## 4. Verify
```cmd
pip install recall-ai
```

Should install from PyPI. Done.

---

## Next session

Pick up here when you return:

1. Go to https://pypi.org/manage/account/token/ and create a **new** API token scoped to **"Entire account"** (not a specific project — that's what caused the 403)
2. Run in the terminal:
   ```cmd
   set TWINE_USERNAME=__token__
   set TWINE_PASSWORD=<paste-your-new-entire-account-token>
   python -m twine upload dist/*
   ```
3. The package is already built as `mnemosyne_memory-0.1.0` in `dist/` — no rebuild needed
4. After a successful upload, verify with `pip install mnemosyne-memory`

**Why it kept failing:**
- First token: invalid/corrupted
- Second token: scoped to project `recall-ai`, which doesn't exist on PyPI yet — project-scoped tokens can't create new projects, only upload to existing ones

---

## Notes
- Build artifacts are already in `dist/` — no need to rebuild unless code changes
- If you do change code before uploading, bump the version in `pyproject.toml` and run `python -m hatch build` again
- Package name: `recall-ai` → installs as `from mnemosyne import Mnemosyne`
