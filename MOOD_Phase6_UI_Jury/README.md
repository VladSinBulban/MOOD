# MOOD Phase 6 — Cleaner UI & Jury Evaluation

This release focuses on two things: reducing clutter in the existing Gradio interface and giving evaluators a repeatable way to test MOOD. It does not add new image-engine features.

## What's changed

- **Editor** — source image, creative direction, collapsed look library, intensity, before/after, status, and export.
- **Adjustments** — exposure, contrast, saturation, temperature, highlights/shadows, sharpness, denoise, optional mask, undo/redo, and saved looks.
- **Batch** — dedicated screen for processing multiple photos.
- **Projects** — dedicated save/load workspace.
- **Jury test** — repeatable test protocol and a feedback form that generates a downloadable text report.
- **Settings** — privacy explanation, release-readiness report, and diagnostics.
- Retains MOOD's black and neon-green `#00FF9A` design and existing image-processing engine.

## Run on Windows

1. Extract this package to a folder such as `D:\Vlad\MOOD`.
2. Keep your existing `.env` file. Do not replace it with `.env.example`.
3. Run `launch_windows.bat`.
4. Open the local URL printed in the terminal (normally `http://127.0.0.1:7862`).

The launcher creates `.venv` if needed and installs dependencies. It only copies `.env.example` to `.env` when `.env` does not already exist.

## Privacy

- The app defaults to local-only access; do not expose the development server directly to the public internet.
- If Gemini is configured, the selected image and prompt are sent to Google for AI editing.
- Local image effects run on the computer.
- Jury feedback is submitted to the MOOD app instance to generate a downloadable text file. It is not sent to a separate analytics service. If the app is hosted online, the host processes the ratings and notes.
- Do not use confidential photos for jury testing.

## Important

This package is a code release, not proof of full end-to-end acceptance. The UI construction and jury feedback generation were tested in the current environment, but the app still needs normal edit, export, two-photo batch, project save/load, and browser testing on the target Windows machine before the jury session.
