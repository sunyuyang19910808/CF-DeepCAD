from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pypandoc
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
SOURCE_MD = BASE_DIR / "论文初稿_Constraint_Fused_DeepCAD_HighModify.md"
WORD_MD = BASE_DIR / "论文初稿_Constraint_Fused_DeepCAD_HighModify_word.md"
DIAGRAM_PNG = BASE_DIR / "论文初稿_Constraint_Fused_DeepCAD_HighModify_arch.png"
OUTPUT_DOCX = BASE_DIR / "论文初稿_Constraint_Fused_DeepCAD_HighModify.docx"
UPDATED_OUTPUT_DOCX = BASE_DIR / "论文初稿_Constraint_Fused_DeepCAD_HighModify_updated.docx"


FORMULA_MAP = {
    "C = C^h ∪ C^v ∪ C^p ∪ C^o": r"C = C^h \cup C^v \cup C^p \cup C^o",
    "z ~ N(0, I)\nS ~ P(S | z)": r"\begin{aligned} z &\sim \mathcal{N}(0, I) \\ S &\sim P(S \mid z) \end{aligned}",
    "Ŝ = D_φ(z),    z = E_θ(S, C)": r"\hat{S} = D_\phi(z), \quad z = E_\theta(S, C)",
    "u_i = [y_i^h, y_i^v] ∈ {0, 1}²": r"u_i = [y_i^h, y_i^v] \in \{0, 1\}^2",
    "p_ij = [y_ij^p, y_ij^o] ∈ {0, 1}²": r"p_{ij} = [y_{ij}^p, y_{ij}^o] \in \{0, 1\}^2",
    "τ_k = (r_k, i_k, j_k)": r"\tau_k = (r_k, i_k, j_k)",
    "e_k^con = LN( W × ( φ(r_k) + F( Concat(λ(i_k), λ(j_k)) ) ) + b ).": r"e_k^{con} = LN\left(W \times \left(\phi(r_k) + F(Concat(\lambda(i_k), \lambda(j_k)))\right) + b\right).",
    "e_t^pre = Ψ_cmd(c_t) + Ψ_arg(a_t) + Ψ_grp(g_t) + ζ(η_t),\ne_t^cad = PE(e_t^pre).": r"\begin{aligned} e_t^{pre} &= \Psi_{cmd}(c_t) + \Psi_{arg}(a_t) + \Psi_{grp}(g_t) + \zeta(\eta_t), \\ e_t^{cad} &= PE(e_t^{pre}). \end{aligned}",
    "X = [e_1^cad, ..., e_T^cad, e_1^con, ..., e_M^con]": r"X = [e_1^{cad}, \ldots, e_T^{cad}, e_1^{con}, \ldots, e_M^{con}]",
    "X": r"X",
    "x̃_n = x_n + s_g(n)": r"\tilde{x}_n = x_n + s_{g(n)}",
    "x_n": r"x_n",
    "1 ≤ n ≤ T": r"1 \le n \le T",
    "T < n ≤ T+M": r"T < n \le T+M",
    "H = TransformerEncoder(X̃)": r"H = TransformerEncoder(\tilde{X})",
    "H^cad = H_1:T\nH^con = H_(T+1):(T+M)": r"\begin{aligned} H^{cad} &= H_{1:T} \\ H^{con} &= H_{(T+1):(T+M)} \end{aligned}",
    "z_cmd = MaskedMean(H^cad)\nz_con = MaskedMean(H^con)\ng = σ(W_g [z_cmd; z_con] + b_g)\nz_mix = g ⊙ z_cmd + (1 - g) ⊙ z_con\nz_pre = W_o [z_cmd; z_con; z_mix] + b_o": r"\begin{aligned} z_{cmd} &= MaskedMean(H^{cad}) \\ z_{con} &= MaskedMean(H^{con}) \\ g &= \sigma(W_g [z_{cmd}; z_{con}] + b_g) \\ z_{mix} &= g \odot z_{cmd} + (1 - g) \odot z_{con} \\ z_{pre} &= W_o [z_{cmd}; z_{con}; z_{mix}] + b_o \end{aligned}",
    "z = B(z_pre)": r"z = B(z_{pre})",
    "z ~ N(0, I),    Ŝ = D_φ(z)": r"z \sim \mathcal{N}(0, I), \quad \hat{S} = D_\phi(z)",
    "G = DecoderLayer(z)": r"G = DecoderLayer(z)",
    "P_cmd = f_cmd(G),\nP_arg = f_arg(G)": r"\begin{aligned} P_{cmd} &= f_{cmd}(G), \\ P_{arg} &= f_{arg}(G) \end{aligned}",
    "G̃ = G + Attention(G, H^con, H^con)": r"\tilde{G} = G + Attention(G, H^{con}, H^{con})",
    "reconstruction: z = E_θ(S, C),    Ŝ = D_φ(z)\ngeneration:     z ~ N(0, I),      Ŝ = D_φ(z)": r"\begin{aligned} \text{reconstruction:}\quad & z = E_\theta(S, C), \quad \hat{S} = D_\phi(z) \\ \text{generation:}\quad & z \sim \mathcal{N}(0, I), \quad \hat{S} = D_\phi(z) \end{aligned}",
    "q_i^dec = Mean({ G_t | line_index_map(t) = i, line_cmd_mask(t) = 1 })": r"q_i^{dec} = Mean(\{G_t \mid line\_index\_map(t) = i,\ line\_cmd\_mask(t) = 1\})",
    "Û_i = f_u(q_i^dec)": r"\hat{U}_i = f_u(q_i^{dec})",
    "m_ij = [q_i^dec; q_j^dec; |q_i^dec - q_j^dec|; q_i^dec ⊙ q_j^dec]": r"m_{ij} = [q_i^{dec}; q_j^{dec}; |q_i^{dec} - q_j^{dec}|; q_i^{dec} \odot q_j^{dec}]",
    "P̂_ij = 1/2 · (f_p(m_ij) + f_p(m_ji))": r"\hat{P}_{ij} = \frac{1}{2} \cdot (f_p(m_{ij}) + f_p(m_{ji}))",
    "L_rec = BCE_w(Û, U) + BCE_w(P̂, P)": r"L_{rec} = BCE_w(\hat{U}, U) + BCE_w(\hat{P}, P)",
    "N_line = Σ_t m_t^line": r"N_{line} = \sum_t m_t^{line}",
    "L_pred = 1 / (N_line · C + ε) · Σ_t m_t^line · Σ_{c=1}^C BCE(ŷ_{t,c}, y_{t,c})": r"L_{pred} = \frac{1}{N_{line} \cdot C + \varepsilon} \cdot \sum_t m_t^{line} \cdot \sum_{c=1}^{C} BCE(\hat{y}_{t,c}, y_{t,c})",
    "d_i = (b_i - a_i) / max(||b_i - a_i||₂, ε)": r"d_i = \frac{b_i - a_i}{\max(\lVert b_i - a_i \rVert_2, \varepsilon)}",
    "r_h(i) = d_{iy}²,    r_v(i) = d_{ix}².": r"r_h(i) = d_{iy}^2, \quad r_v(i) = d_{ix}^2.",
    "r_p(i, j) = 1 - |d_i · d_j|,    r_o(i, j) = |d_i · d_j|.": r"r_p(i, j) = 1 - |d_i \cdot d_j|, \quad r_o(i, j) = |d_i \cdot d_j|.",
    "L_geom =\n1 / max(N_geom, 1) · (\n  Σ_{i: m_i u_i^h = 1} r_h(i)\n  + Σ_{i: m_i u_i^v = 1} r_v(i)\n  + Σ_{(i,j): m_i m_j p_{ij} = 1} r_p(i, j)\n  + Σ_{(i,j): m_i m_j o_{ij} = 1} r_o(i, j)\n).": r"\begin{aligned} L_{geom} = \frac{1}{\max(N_{geom}, 1)} \cdot (&\sum_{i: m_i u_i^h = 1} r_h(i) \\ &+ \sum_{i: m_i u_i^v = 1} r_v(i) \\ &+ \sum_{(i,j): m_i m_j p_{ij} = 1} r_p(i, j) \\ &+ \sum_{(i,j): m_i m_j o_{ij} = 1} r_o(i, j)). \end{aligned}",
    "L = L_cmd + alpha · L_pred + beta · L_rec + gamma · L_geom": r"L = L_{cmd} + \alpha \cdot L_{pred} + \beta \cdot L_{rec} + \gamma \cdot L_{geom}",
    "L = L_cmd + alpha · L_pred（可选，与ζ(η_t)状态同） + beta · L_rec + gamma · L_geom": r"L = L_{cmd} + \alpha \cdot L_{pred}\;(\text{optional, same status as } \zeta(\eta_t)) + \beta \cdot L_{rec} + \gamma \cdot L_{geom}",
}

INLINE_FORMULA_MAP = {
    "S ~ P(S | z)": r"S \sim P(S \mid z)",
    "z ~ N(0, I)": r"z \sim \mathcal{N}(0, I)",
    "P(S | z, C)": r"P(S \mid z, C)",
    "P(S | z)": r"P(S \mid z)",
    "P(S | z, H^con)": r"P(S \mid z, H^{con})",
    "S = {(c_t, a_t)}_{t=1}^T": r"S = \{(c_t, a_t)\}_{t=1}^{T}",
    "S": r"S",
    "c_t": r"c_t",
    "a_t": r"a_t",
    "L = {l_i}_{i=1}^N": r"L = \{l_i\}_{i=1}^{N}",
    "C^h": r"C^h",
    "C^v": r"C^v",
    "C^p": r"C^p",
    "C^o": r"C^o",
    "E_θ": r"E_\theta",
    "D_φ": r"D_\phi",
    "C": r"C",
    "z": r"z",
    "X": r"X",
    "x_n": r"x_n",
    "1 ≤ n ≤ T": r"1 \le n \le T",
    "T < n ≤ T+M": r"T < n \le T+M",
    "Concat(u, v)": r"Concat(u, v)",
    "Concat(u, v)": r"Concat(u, v)",
    "u": r"u",
    "v": r"v",
    "[u; v]": r"[u; v]",
    "Attention(Q, K, V)": r"Attention(Q, K, V)",
    "MultiHead(·)": r"MultiHead(\cdot)",
    "Dropout(·)": r"Dropout(\cdot)",
    "LN(·)": r"LN(\cdot)",
    "softmax(Q K^T / sqrt(d_k)) V": r"softmax(QK^T / \sqrt{d_k})V",
    "d_k": r"d_k",
    "ζ(·)": r"\zeta(\cdot)",
    "η_t ∈ R^4": r"\eta_t \in R^4",
    "R^d": r"R^d",
    "y_i^h = 1": r"y_i^h = 1",
    "y_i^v = 1": r"y_i^v = 1",
    "(i, j)": r"(i, j)",
    "(l_i, l_j)": r"(l_i, l_j)",
    "y_ij^p = 1": r"y_{ij}^p = 1",
    "y_ij^o = 1": r"y_{ij}^o = 1",
    "l_i": r"l_i",
    "l_j": r"l_j",
    "p_ij = p_ji": r"p_{ij} = p_{ji}",
    "r_k": r"r_k",
    "i_k": r"i_k",
    "j_k": r"j_k",
    "τ_k": r"\tau_k",
    "M_max = 128": r"M_{max} = 128",
    "M_max": r"M_{max}",
    "M": r"M",
    "e_k^con ∈ R^d": r"e_k^{con} \in R^d",
    "φ(·)": r"\phi(\cdot)",
    "λ(·)": r"\lambda(\cdot)",
    "R^{d_l}": r"R^{d_l}",
    "F: R^{2d_l} → R^d": r"F: R^{2d_l} \to R^d",
    "F": r"F",
    "k": r"k",
    "W ∈ R^{d×d}": r"W \in R^{d \times d}",
    "b ∈ R^d": r"b \in R^d",
    "i_k = j_k": r"i_k = j_k",
    "j_k = i_k": r"j_k = i_k",
    "Concat(λ(i_k), λ(j_k))": r"Concat(\lambda(i_k), \lambda(j_k))",
    "t": r"t",
    "Ψ_cmd(c_t)": r"\Psi_{cmd}(c_t)",
    "Ψ_arg(a_t)": r"\Psi_{arg}(a_t)",
    "Ψ_grp(g_t)": r"\Psi_{grp}(g_t)",
    "ζ(η_t)": r"\zeta(\eta_t)",
    "η_t ∈ R^4": r"\eta_t \in R^4",
    "η_t": r"\eta_t",
    "e_t^cad": r"e_t^{cad}",
    "e_k^con": r"e_k^{con}",
    "g(n) ∈ {cad, con}": r"g(n) \in \{cad, con\}",
    "g(n) ∈ {0, 1}": r"g(n) \in \{0, 1\}",
    "g(n)=0": r"g(n)=0",
    "g(n)=1": r"g(n)=1",
    "s_g(n)": r"s_{g(n)}",
    "s_0": r"s_0",
    "s_1": r"s_1",
    "T=60": r"T=60",
    "M=8": r"M=8",
    "n": r"n",
    "H^cad": r"H^{cad}",
    "H^con": r"H^{con}",
    "z_cmd": r"z_{cmd}",
    "z_con": r"z_{con}",
    "z_mix": r"z_{mix}",
    "[z_cmd; z_con]": r"[z_{cmd}; z_{con}]",
    "[z_cmd; z_con; z_mix]": r"[z_{cmd}; z_{con}; z_{mix}]",
    "z_cmd, z_con": r"z_{cmd}, z_{con}",
    "W_g": r"W_g",
    "b_g": r"b_g",
    "W_o": r"W_o",
    "b_o": r"b_o",
    "σ(·)": r"\sigma(\cdot)",
    "[0,1]": r"[0,1]",
    "g": r"g",
    "⊙": r"\odot",
    "z_mix = g ⊙ z_cmd + (1 - g) ⊙ z_con": r"z_{mix} = g \odot z_{cmd} + (1 - g) \odot z_{con}",
    "z_pre": r"z_{pre}",
    "1 × 256": r"1 \times 256",
    "1 × 512": r"1 \times 512",
    "z ∈ R^{1×B×512}": r"z \in R^{1 \times B \times 512}",
    "P_cmd": r"P_{cmd}",
    "P_arg": r"P_{arg}",
    "G ∈ R^{B×T×d}": r"G \in R^{B \times T \times d}",
    "i": r"i",
    "u_i = [y_i^h, y_i^v]": r"u_i = [y_i^h, y_i^v]",
    "Û": r"\hat{U}",
    "U": r"U",
    "BCE_w": r"BCE_w",
    "q_i^dec": r"q_i^{dec}",
    "L_rec": r"L_{rec}",
    "m_t^line ∈ {0, 1}": r"m_t^{line} \in \{0, 1\}",
    "ŷ_t ∈ [0, 1]^C": r"\hat{y}_t \in [0, 1]^C",
    "y_t ∈ {0, 1}^C": r"y_t \in \{0, 1\}^C",
    "C = 4": r"C = 4",
    "a_i, b_i ∈ R²": r"a_i, b_i \in R^2",
    "d_{iy} = 0": r"d_{iy} = 0",
    "d_{ix} = 0": r"d_{ix} = 0",
    "|d_i · d_j| → 1": r"|d_i \cdot d_j| \to 1",
    "|d_i · d_j| → 0": r"|d_i \cdot d_j| \to 0",
    "m_i ∈ {0,1}": r"m_i \in \{0, 1\}",
    "u_i^h": r"u_i^h",
    "u_i^v": r"u_i^v",
    "p_{ij}": r"p_{ij}",
    "o_{ij}": r"o_{ij}",
    "N_geom": r"N_{geom}",
    "L_cmd": r"L_{cmd}",
    "L_pred": r"L_{pred}",
    "L_geom": r"L_{geom}",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "ACC_cmd": r"ACC_{cmd}",
    "ACC_param": r"ACC_{param}",
    "z=512": r"z = 512",
    "L": r"L",
    "G": r"G",
}

CODE_IDENTIFIERS = {
    "constraint_tags",
    "constraint_memory",
    "line_cmd_mask",
    "line_index_map",
    "decoder-side",
    "latent-only",
    "full-sequence",
    "masked",
    "mean",
    "pooling",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for item in candidates:
        if item.exists():
            return ImageFont.truetype(str(item), size=size)
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.FreeTypeFont) -> None:
    wrapped = "\n".join(textwrap.wrap(text, width=14, break_long_words=False))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=fnt, spacing=4, align="center")
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - w) / 2
    y = box[1] + (box[3] - box[1] - h) / 2
    draw.multiline_text((x, y), wrapped, fill=(30, 30, 30), font=fnt, spacing=4, align="center")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=(60, 60, 60), width=3)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - direction * 12, ey - 7), (ex - direction * 12, ey + 7)]
    else:
        direction = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 7, ey - direction * 12), (ex + 7, ey - direction * 12)]
    draw.polygon(points, fill=(60, 60, 60))


def render_diagram() -> None:
    img = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(img)
    box_font = font(30)
    title_font = font(42, bold=True)
    draw.text((60, 35), "CF-DeepCAD 总体流程", fill=(20, 20, 20), font=title_font)

    boxes = {
        "cad": (80, 150, 330, 240, "CAD命令参数序列"),
        "con": (80, 310, 330, 400, "几何约束Token"),
        "enc": (430, 230, 710, 330, "约束融合编码器"),
        "cmd": (800, 130, 1050, 220, "命令记忆"),
        "cmem": (800, 330, 1050, 420, "约束记忆"),
        "pool": (1140, 230, 1430, 330, "命令/约束分离池化"),
        "z": (1510, 230, 1740, 330, "扩容潜变量 z"),
        "dec": (1510, 460, 1740, 560, "Latent-only 解码器"),
        "seq": (1210, 610, 1460, 700, "命令与参数预测"),
        "hid": (1510, 610, 1740, 700, "Decoder hidden states"),
        "tag": (1210, 770, 1460, 850, "线级约束预测"),
        "rec": (1510, 770, 1740, 850, "unary/pair 约束重建"),
        "geom": (880, 610, 1130, 700, "可微草图解释"),
    }

    for box in boxes.values():
        x1, y1, x2, y2, label = box
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, outline=(70, 100, 150), width=3, fill=(245, 248, 252))
        draw_centered(draw, (x1, y1, x2, y2), label, box_font)

    arrow(draw, (330, 195), (430, 260))
    arrow(draw, (330, 355), (430, 300))
    arrow(draw, (710, 260), (800, 175))
    arrow(draw, (710, 300), (800, 375))
    arrow(draw, (1050, 175), (1140, 260))
    arrow(draw, (1050, 375), (1140, 300))
    arrow(draw, (1430, 280), (1510, 280))
    arrow(draw, (1625, 330), (1625, 460))
    arrow(draw, (1510, 520), (1460, 650))
    arrow(draw, (1625, 560), (1625, 610))
    arrow(draw, (1510, 660), (1460, 810))
    arrow(draw, (1625, 700), (1625, 770))
    arrow(draw, (1210, 650), (1130, 650))

    img.save(DIAGRAM_PNG)


def normalize_markdown(md: str) -> str:
    def replace_mermaid(match: re.Match[str]) -> str:
        return f"![CF-DeepCAD 总体流程]({DIAGRAM_PNG.name})"

    def replace_text_formula(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        formula = FORMULA_MAP.get(content)
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

    md = re.sub(r"```mermaid\n.*?\n```", replace_mermaid, md, flags=re.S)
    md = re.sub(r"```text\n(.*?)\n```", replace_text_formula, md, flags=re.S)
    md = re.sub(r"`([^`\n]+)`", replace_inline_formula, md)
    return md


def main() -> None:
    render_diagram()
    source = SOURCE_MD.read_text(encoding="utf-8")
    normalized = normalize_markdown(source)
    WORD_MD.write_text(normalized.replace("\r\n", "\n"), encoding="utf-8")
    output_docx = OUTPUT_DOCX
    try:
        pypandoc.convert_file(
            str(WORD_MD),
            "docx",
            outputfile=str(output_docx),
            extra_args=[
                "--standalone",
                f"--resource-path={BASE_DIR}",
            ],
        )
    except RuntimeError as exc:
        if "permission denied" not in str(exc).lower():
            raise
        output_docx = UPDATED_OUTPUT_DOCX
        pypandoc.convert_file(
            str(WORD_MD),
            "docx",
            outputfile=str(output_docx),
            extra_args=[
                "--standalone",
                f"--resource-path={BASE_DIR}",
            ],
        )
    print(f"Generated: {WORD_MD}")
    print(f"Generated: {DIAGRAM_PNG}")
    print(f"Generated: {output_docx}")


if __name__ == "__main__":
    main()
