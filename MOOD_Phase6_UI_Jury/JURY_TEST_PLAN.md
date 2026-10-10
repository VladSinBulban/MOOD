# MOOD Jury Test Plan

## Purpose

Evaluate the current MOOD build using the same tasks in the same order. Record actual outcomes, not assumptions.

## Before the session

- Start MOOD using `launch_windows.bat`.
- Use two non-sensitive sample photos in JPG or PNG format.
- Confirm the app opens in the browser.
- If Gemini is configured, confirm the jury understands that image and prompt data may be sent to Google. If Gemini is not configured, local effects are conventional image processing rather than generative editing.

## Task 1 — Single-photo edit

1. Open **Editor**.
2. Upload a sample photo.
3. Enter a simple prompt such as `warm natural editorial color, preserve skin tones`.
4. Click **Transform photo**.
5. Record whether the action completes, whether the status is clear, and whether the output matches the request.

**Pass criteria:** a result is displayed or a clear failure message is provided; the app remains usable.

## Task 2 — Compare and export

1. Compare the original and edited previews.
2. Export as PNG.
3. Download and open the file.

**Pass criteria:** the downloaded file opens and contains the edited result.

## Task 3 — Batch of two photos

1. Open **Batch**.
2. Select exactly two sample photos.
3. Enter one batch direction and start processing.
4. Download the ZIP and inspect its contents.

**Pass criteria:** the job completes or reports a clear failure; the ZIP includes a result for each successfully processed input and an explanatory manifest where supported.

## Task 4 — Adjustments and project workflow

1. Open **Adjustments**, change one control, and apply it.
2. Try Undo and Redo.
3. Open **Projects**, save a project ZIP, and optionally load it again.

**Pass criteria:** actions complete without crashing and status messages make the outcome understandable.

## Record feedback

Open **Jury test** inside MOOD. Complete the rating fields and describe confusing steps, errors, or missing information. Create and download the feedback report, then send it to the project owner.

## Do not collect

Do not ask jurors to enter passwords, API keys, private images, or unnecessary personal information.
