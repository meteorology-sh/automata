# Virtual Environment

The venv lives at `venv/` in the project root; MuJoCo, torch and all other dependencies are
installed there. Call its interpreter directly rather than relying on an activated shell, since
shell state does not persist between commands:

```bash
venv/bin/python main.py --config configs/default.yaml     # Linux / macOS
venv/Scripts/python main.py --config configs/default.yaml  # Windows
```

Create it with `python3.12 -m venv venv && venv/bin/pip install -r requirements.txt`.
