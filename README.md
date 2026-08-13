# Project Aether

An AI-first operating system. A single continuous entity is the interface —
no chat threads, no sessions, no app launcher, no windows. You talk to it.

## Status

Early. Building the input layer.

## Architecture

**Input** — fully local, CPU-only, no LLM. Voice activity detection gates a
streaming speech recognizer. In parallel, signal processing measures pitch,
loudness, duration, and pauses from the raw audio. These are aligned on word
timestamps to produce a transcript where every word carries a weight — how
much it was emphasized, how it was said. Continuous values are kept
internally; a compact marker notation is used at the boundary to any
language model.

**Memory** — a retrieval ladder. Consolidated gist first, covering most
requests. Transcript search when that is insufficient. Audio and video only
on explicit request. Raw history is kept complete and is read by the system,
never surfaced as a scrollback. Consolidation always runs from raw source so
summaries never compound drift. Memories decay in weight rather than being
deleted.

**Storage** — no hierarchy. Content-addressed blobs in flat storage; all
meaning lives in a database. Groups are queries, not folders.

**Models** — swappable behind a capability contract. The system requests a
capability, never a named model. Prompts are generated from task and model
manifest, never hand-tuned per model.

## Principles

- Speech recognition and signal processing are categorically different in
  cost from language model calls. Nothing in the input path calls a model.
- Continuous measurements are preserved internally even where compact
  markers are used at boundaries.
- The raw archive is inviolable. Anything derived or reconstructed is marked
  as such and never replaces the original.
- Most answers are one spoken line. A visual has to earn its place.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, on macOS, Homebrew.

    brew install portaudio libsndfile ffmpeg
    uv sync

## Current scripts

`record.py` captures microphone audio. `transcribe.py` runs speech
recognition with word timestamps. `prosody_test.py` measures pitch,
loudness, duration, and pauses per word. `plot_prosody.py` draws the
contours with word boundaries marked.