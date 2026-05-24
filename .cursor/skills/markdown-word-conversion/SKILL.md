---
name: markdown-word-conversion
description: Convert Markdown papers or technical documents to Word .docx with editable formulas, tables, and diagrams. Use when the user asks to convert Markdown to Word, preserve formulas/subscripts/superscripts, handle Mermaid diagrams, or prepare a paper draft for Word editing.
---

# Markdown Word Conversion

## When To Use

Use this skill when converting Markdown papers or technical documents into Word, especially when the document contains:

- LaTeX-like formulas, inline symbols, subscripts, superscripts, Greek letters, or math operators.
- Markdown tables that should remain editable in Word.
- Mermaid diagrams that should be embedded as images.
- A requirement to keep the original Markdown source unchanged.

## Core Workflow

1. Keep the original Markdown file unchanged.
2. Create a conversion copy next to the source, usually named `<stem>_word.md`.
3. Convert mathematical content before running Pandoc:
   - Convert formula code blocks marked as `text` into display math blocks such as `$$...$$` when they contain equations.
   - Convert inline math wrapped in backticks, such as `S = {(c_t, a_t)}_{t=1}^T`, into `$S = \{(c_t, a_t)\}_{t=1}^{T}$`.
   - Keep real code identifiers such as `constraint_memory`, `line_cmd_mask`, file names, commands, and config keys as code.
4. Render Mermaid diagrams to images and replace Mermaid blocks with Markdown image links.
5. Use Pandoc to generate `.docx`.
6. Validate the generated `.docx` by inspecting the Word package:
   - Count `<m:oMath` objects to confirm formulas were converted.
   - Confirm expected images exist under `word/media/`.
   - Search for literal math leftovers such as `^T`, `c_t`, `ACC_cmd`, or other raw source forms.

## Recommended Tooling

Prefer Pandoc for Word output because it can produce editable Word equation objects from Markdown math.

If `pandoc` is not on `PATH`, use `pypandoc`:

```powershell
python -m pip install pypandoc
python -c "import pypandoc; pypandoc.download_pandoc(); print(pypandoc.get_pandoc_path())"
```

For diagrams:

- First try `mmdc` or `npx -y @mermaid-js/mermaid-cli`.
- If Mermaid CLI is unavailable or hangs, generate an equivalent PNG with Python/PIL and embed that image.

## Formula Conversion Rules

Treat these as math and convert to `$...$` or `$$...$$`:

- Variables with subscripts or superscripts: `c_t`, `C^h`, `H^con`, `ACC_cmd`.
- Full formulas: `S = {(c_t, a_t)}_{t=1}^T`, `z ~ N(0, I)`, `P(S | z)`.
- Greek-symbol expressions: `E_θ`, `D_φ`, `ζ(η_t)`, `τ_k`.
- Set and vector expressions: `{0, 1}²`, `R^d`, `R^{B×T×d}`.
- Loss names and metrics: `L_cmd`, `L_pred`, `L_rec`, `L_geom`, `ACC_param`.

Keep these as code unless the user explicitly wants them formatted as math:

- Program fields: `constraint_memory`, `constraint_tags`, `line_cmd_mask`, `line_index_map`.
- File paths, commands, config keys, package names, and script names.
- Literal method names where code styling is clearer than math styling.

## Pandoc Command

Use this pattern:

```powershell
python path/to/convert_markdown_to_word.py
```

Or directly:

```powershell
pandoc "<source_word.md>" -o "<output.docx>" --standalone --resource-path="<document_directory>"
```

## Validation Snippet

After generating Word, inspect the `.docx` package:

```powershell
python -c "from pathlib import Path; import zipfile; p=Path(r'<output.docx>'); z=zipfile.ZipFile(p); doc=z.read('word/document.xml').decode('utf-8'); print('bytes', p.stat().st_size); print('omath', doc.count('<m:oMath'), 'pictures', doc.count('<w:drawing')); print('literal ^T', doc.count('^T'), 'literal c_t', doc.count('c_t'), 'literal ACC_cmd', doc.count('ACC_cmd')); print('has superscript', '<m:sSup>' in doc, 'has subscript', '<m:sSub>' in doc, 'has frac', '<m:f>' in doc)"
```

Good signs:

- `<m:oMath` count is nonzero and increases after inline formula conversion.
- `<m:sSup>` and `<m:sSub>` are present for superscripts and subscripts.
- Literal leftovers such as `^T`, `c_t`, `ACC_cmd` are zero unless they are intentionally code.

## Project Example

The conversion used for the CF-DeepCAD paper is implemented at:

`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/convert_markdown_to_word.py`

Use it as a concrete reference for:

- Formula block mapping.
- Inline formula mapping.
- Preserving code identifiers.
- Rendering a fallback architecture diagram with PIL.
- Verifying Word math objects after conversion.
