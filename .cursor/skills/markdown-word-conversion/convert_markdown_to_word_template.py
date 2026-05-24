from __future__ import annotations

import re
from pathlib import Path

import pypandoc


SOURCE_MD = Path("paper.md")
WORD_MD = SOURCE_MD.with_name(f"{SOURCE_MD.stem}_word.md")
OUTPUT_DOCX = SOURCE_MD.with_suffix(".docx")

# Fill these maps from the paper being converted.
DISPLAY_FORMULA_MAP = {
    "z ~ N(0, I)\nS ~ P(S | z)": r"\begin{aligned} z &\sim \mathcal{N}(0, I) \\ S &\sim P(S \mid z) \end{aligned}",
}

INLINE_FORMULA_MAP = {
    "S = {(c_t, a_t)}_{t=1}^T": r"S = \{(c_t, a_t)\}_{t=1}^{T}",
    "c_t": r"c_t",
    "ACC_cmd": r"ACC_{cmd}",
}

CODE_IDENTIFIERS = {
    "constraint_memory",
    "line_cmd_mask",
    "line_index_map",
}


def normalize_markdown(md: str) -> str:
    def replace_text_formula(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        formula = DISPLAY_FORMULA_MAP.get(content)
        if not formula:
            return match.group(0)
        return f"$$\n{formula}\n$$"

    def replace_inline_formula(match: re.Match[str]) -> str:
        content = match.group(1)
        if content in CODE_IDENTIFIERS:
            return match.group(0)
        formula = INLINE_FORMULA_MAP.get(content)
        if formula:
            return f"${formula}$"
        return match.group(0)

    md = re.sub(r"```text\n(.*?)\n```", replace_text_formula, md, flags=re.S)
    md = re.sub(r"`([^`\n]+)`", replace_inline_formula, md)
    return md


def main() -> None:
    source = SOURCE_MD.read_text(encoding="utf-8")
    WORD_MD.write_text(normalize_markdown(source).replace("\r\n", "\n"), encoding="utf-8")
    pypandoc.convert_file(
        str(WORD_MD),
        "docx",
        outputfile=str(OUTPUT_DOCX),
        extra_args=["--standalone", f"--resource-path={SOURCE_MD.parent}"],
    )
    print(f"Generated: {WORD_MD}")
    print(f"Generated: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
