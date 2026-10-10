# MOOD local scenario engine update

This package adds a JSON-driven local editing engine to the existing MOOD project.

- `scenarios.json`: 151 curated and derived scenario recipes, aliases, color-grade settings, and atmospheric effects.
- `src/image_engine.py`: prompt matching, recipe composition, and local effects.
- `apply_update.ps1`: safely replaces the old local-editing section in the existing `app.py` with an import of the new engine. It makes a timestamped backup in `%TEMP%`.

## Install

1. Extract this ZIP into the existing `D:\Vlad\MOOD` folder, allowing it to merge/overwrite `src/image_engine.py` and `src/__init__.py`, and add `scenarios.json` and `apply_update.ps1`.
2. In the VS Code terminal, from `D:\Vlad\MOOD`, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\apply_update.ps1
   ```

3. Run the test command:

   ```powershell
   python -m unittest discover -s tests -v
   ```

4. Start the app with your existing run command (usually `python app.py`).

No paid API, hosting, or new Python package is needed for the local fallback. The Gemini attempt remains as it was. If Gemini fails, the fallback now composes JSON recipes and local effects. It is still conventional image processing, not generative AI; it will not reliably add realistic new objects or replace scenery.
