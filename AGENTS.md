# AI Video Learning Workspace — Agent Instructions

## 1. Product and scope

Build a maintainable application that helps non-native English speakers **learn technical subjects through English-language videos** ("Learn through English, not Learn English"). Initial audience: Japanese STEM students, researchers, and early-career engineers, especially in AI/IT/data science.

- User-facing UI text: natural Japanese by default. Preserve the original English video transcript; do not translate it unless asked.
- Core experience: local video + timestamped English transcript; later, sentence-based video seeking and separate **Language / Knowledge** explanations grounded in the video context.
- Streamlit is a **prototype UI**, not a commitment to the final desktop stack. Keep reusable logic independent of Streamlit.
- Prefer local processing and user-controlled files. No cloud upload, URL downloader, Anki integration, or other product expansion unless requested.

## 2. Source of truth and current boundaries

**Read the real repository before acting.** The list below describes responsibilities, not a promise that every feature is finished. Actual code, tests, and the current task take precedence over stale status descriptions.

| Area | Responsibility |
| --- | --- |
| `app.py` | Streamlit presentation, user events and UI state; not core processing |
| `src/asr.py` | CUDA setup, faster-whisper model loading and transcription |
| `src/transcript.py` | Versioned ASR transcript schema and JSON serialization |
| `src/video_loader.py` | Local video import, content ID, deduplication, safe storage |
| `src/video_upload.py` | Adapter from Streamlit uploaded files to local import |
| `src/video_transcript.py` | Video discovery, transcript lookup and UI-facing validation |
| `src/playback.py` | Media probing and optional compatible playback proxy |
| `src/transcription_service.py` | **Proposed** ASR workflow orchestration; verify existence before use |
| `notebooks/` | Exploration and benchmarks; move stable reusable code to `src/` |
| `tests/` | Fast deterministic tests plus explicitly separated integration tests |

Do not invent missing files or assume a planned feature has already been implemented. Preserve existing working interfaces unless the task requires a deliberate change.

## 3. Engineering rules

1. Make the **smallest coherent, maintainable change** that solves the requested problem. Avoid hacks, duplicated logic, unnecessary layers and speculative extensibility.
2. Read relevant implementation, tests and configuration before editing. Follow existing conventions and dependencies.
3. Keep module boundaries: UI triggers actions; application services orchestrate; core modules implement processing and persistence. Core modules must not import Streamlit.
4. Use explicit functions, useful type hints, concise docstrings, and `pathlib.Path`. Avoid hard-coded machine-specific paths and unnecessary classes.
5. Keep the original video, ASR transcript and derived playback cache separate. A playback proxy must not change the original video's identity or transcript association.
6. Use stable IDs and versioned data formats where already established. Do not silently overwrite user data or change schema semantics.
7. Handle missing files/tools, permission errors, incomplete writes and reruns predictably. Do not hide failures with silent fallbacks.
8. Before a breaking API/schema change or major architectural change, explain trade-offs and **ask for approval**. Otherwise implement within the task's scope without unnecessary confirmation.
9. Respect user-specified file scopes. Do not change model parameters, unrelated modules or notebooks as incidental cleanup.

## 4. ASR and learning data

- Validated ASR baseline: `faster-whisper`, model `turbo`, language `en`, CUDA, `float16`, `beam_size=1`, `word_timestamps=True`. Do not change without explicit instruction or benchmark evidence.
- ASR operates on the **original video**, not the H.264 playback copy.
- Transcript JSON is versioned and contains `schema_version`, `video`, `asr`, `segments`; segments retain start/end/text and word-level timing where available.
- Keep translations, Japanese readings (furigana), Language explanations and Knowledge explanations in separate, linked data rather than overwriting the source transcript.
- Importing a video, requesting transcription and displaying a transcript are separate operations. Any automatic transcription is an **application-level trigger**, never a side effect inside `video_loader.py`. Streamlit reruns must not cause unintended repeated inference.

## 5. Model and local data storage

- Never commit model weights, user videos, generated transcripts, playback caches, credentials or personal data to Git. Check `.gitignore` before staging files.
- Do not embed large model weights inside the source tree or require one model copy per project. Prefer runtime-supported shared caches (e.g. Hugging Face Hub); allow configurable locations in the future desktop app.
- Distinguish **model download**, **model loading**, and **model inference**. Reuse existing downloads when possible; do not download or install models/tools without the task requiring it.
- Paths must be configurable and OS-aware. Never hard-code this developer's `D:` drive as an application default.
- Preserve original imported videos and user learning data. Generated playback proxies and temporary uploads belong in separately managed caches; failures must not corrupt originals.
- The final desktop UI should offer user-friendly local file selection and model storage settings. Do not build these into the Streamlit prototype unless requested.

## 6. Validation and delivery

- Add or update tests for behavioral changes; prefer deterministic, fast tests using temporary directories and mocks.
- Run relevant tests, then the full **default** `python -m pytest` suite when practical. Keep GPU ASR and external-tool (FFmpeg) integration tests explicitly opt-in or conditional; never misreport a skip as a pass.
- When changing UI, manually verify key interactions if the environment permits; otherwise state exactly what still needs local verification.
- Do not claim success merely because code was written. Report: files changed, behavior, commands/tests executed and results, skipped checks, and remaining risks.
- Git commits are separate from coding: inspect changed files first, never commit ignored/private files, and do not create commits or push without instruction.

## 7. Working protocol for coding agents

1. **Inspect** the requested area and existing tests; identify dependencies and any conflicts with these instructions.
2. **Plan briefly** only when the change spans modules, alters interfaces, or carries material risk. If the user requests review-only, do not edit.
3. **Implement** the smallest scoped solution; preserve backward behavior unless explicitly changed.
4. **Verify** with targeted tests and, where appropriate, the default suite; distinguish mock tests from real GPU/media validation.
5. **Summarize** changes and evidence concisely. Explain important commands, their purpose, and why they were run; do not merely list commands.

# Development and Learning Guidelines

This project has two equally important objectives:

1. Build a maintainable, functional software product.
2. Improve the developer's independent software engineering skills.

## Before implementation

- For unfamiliar technologies, briefly explain the essential concepts needed for the task.
- Identify the relevant components, files, and interfaces.
- For nontrivial changes, explain the proposed approach before implementation.
- Do not introduce unnecessary abstractions, frameworks, or dependencies.

## During implementation

- Prefer small, focused, testable changes.
- Follow existing project conventions.
- Use established libraries rather than reimplementing standard functionality.
- Do not sacrifice code quality merely to simplify learning.
- Add appropriate tests for behavioral changes.

## After implementation

Provide a concise explanation covering:

1. Which files changed and why.
2. The key execution path or data flow.
3. Which concepts the developer should understand.
4. How the implementation was verified.
5. One small follow-up modification the developer can attempt independently.

## Learning assistance

- When the developer asks for an explanation, prioritize reasoning and concrete examples over large code dumps.
- When the developer asks for a hint, do not immediately reveal the full implementation.
- When the developer requests implementation, complete the requested work rather than unnecessarily withholding code.
- Clearly distinguish verified behavior from assumptions.

The goal is not to avoid AI-assisted coding, but to ensure that AI-assisted development also produces transferable engineering knowledge.

For a specific user request, its explicit scope and acceptance criteria override suggestions or examples in this file. If instructions conflict with safety or data integrity, stop and explain the conflict.
