# Dynamic Resume Keyword Highlighter

A single-script tool that **reads your LaTeX resume**, **parses a job description**, dynamically **matches skills (no hardcoded lists)**, and **bolds matched keywords** in your LaTeX files. It optionally compiles to PDF.

> Works without maintaining a giant skills list—uses heuristics, abbreviation detection (e.g., `GCP` ⇄ `Google Cloud Platform`), fuzzy matching, and (optionally) semantic similarity (SBERT).

---

## Features

- Extracts skills from LaTeX sections (`experience.tex`, `projects.tex`, `skills.tex`, `education.tex`)
- Matches against a JD via exact + fuzzy + optional semantic (SBERT) similarity
- Smart abbreviation mapping near mentions (e.g., `AI` ⇄ `Artificial Intelligence`)
- Avoids false positives (`learn`, degree headers, GPA lines, etc.)
- Bold-inserts only in target `.tex` files (protects `\href{}`, already-bold text, URLs)
- Caches external skill sources for speed

---

## Project layout

