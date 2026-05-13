#!/usr/bin/env python3
"""Render per-layout PNG diagrams for the default mainloop_tma_ws layouts."""

from __future__ import annotations

import argparse
import colorsys
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "layout_visualizations"

TILER_PNG = OUTPUT_DIR / "mainloop_tiler_composition.png"
THREAD_PNG = OUTPUT_DIR / "thread_layout_warp_mapping.png"
FRAG_A_PNG = OUTPUT_DIR / "atom_layout_a_tv.png"
FRAG_B_PNG = OUTPUT_DIR / "atom_layout_b_tv.png"
FRAG_C_PNG = OUTPUT_DIR / "atom_layout_c_tv.png"
SMEM_Q_PNG = OUTPUT_DIR / "smem_layout_q_thread_access.png"
SMEM_K_PNG = OUTPUT_DIR / "smem_layout_k_thread_access.png"
SMEM_VT_PNG = OUTPUT_DIR / "smem_layout_vt_thread_access.png"

TILE_M = 128
TILE_N = 128
ATOM_M = 16
ATOM_N = 32
ATOM_K = 64
M_GROUPS = 8
LANES = 32
STAGES = 3


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = font(30, True)
FONT_H2 = font(22, True)
FONT_H3 = font(16, True)
FONT = font(14)
FONT_SMALL = font(11)
FONT_TINY = font(9)


def thread_color(thread: int) -> tuple[int, int, int]:
    hue = (thread * 0.61803398875) % 1.0
    sat = 0.62
    val = 0.90 if (thread // LANES) % 2 == 0 else 0.72
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return tuple(int(c * 255) for c in rgb)


def atom_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.137 + 0.58) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.48, 0.86)
    return tuple(int(c * 255) for c in rgb)


def contrast_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 24, 32) if luminance > 155 else (255, 255, 255)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
              fill: tuple[int, int, int] = (25, 31, 42),
              fnt: ImageFont.ImageFont = FONT) -> None:
    draw.text(xy, text, fill=fill, font=fnt)


def rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
         fill: tuple[int, int, int], outline: tuple[int, int, int] = (180, 188, 202),
         width: int = 1) -> None:
    draw.rectangle(box, fill=fill, outline=outline, width=width)


def new_canvas(width: int, height: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw_text(draw, (34, 28), title, fnt=FONT_TITLE)
    draw_text(draw, (34, 68), subtitle, fill=(78, 86, 100))
    return img, draw


def render_tiler_composition() -> None:
    img, draw = new_canvas(
        980,
        760,
        "mainloop_tma_ws tiler composition",
        "TileShape=(M128,N128,K128), MMA atom=(M16,N32,K64), AtomLayoutMNK=(8,1,1)",
    )

    left, top = 64, 130
    cell_w, cell_h = 72, 48
    draw_text(draw, (left, top - 36), "Full M x N block: 8 atom rows x 4 N slices", fnt=FONT_H2)
    draw_text(draw, (left, top - 10), "Each rectangle is one 16x32 atom footprint. K has two 64-deep phases.", fill=(82, 90, 105), fnt=FONT_SMALL)

    for m_group in range(M_GROUPS):
        y0 = top + m_group * cell_h
        draw_text(draw, (left - 44, y0 + 15), f"M{m_group}", fill=(77, 85, 99), fnt=FONT_SMALL)
        for n_slice in range(4):
            x0 = left + n_slice * cell_w
            atom = m_group * 4 + n_slice
            rect(draw, (x0, y0, x0 + cell_w, y0 + cell_h), atom_color(atom), (255, 255, 255), 2)
            draw_text(draw, (x0 + 12, y0 + 13), f"A{atom}", fill=(255, 255, 255), fnt=FONT_H3)
            draw_text(draw, (x0 + 9, y0 + 30), f"{m_group},{n_slice}", fill=(245, 248, 255), fnt=FONT_SMALL)

    for n_slice in range(4):
        x = left + n_slice * cell_w + 10
        draw_text(draw, (x, top + M_GROUPS * cell_h + 8), f"N{n_slice}\n{n_slice * 32}-{n_slice * 32 + 31}", fill=(77, 85, 99), fnt=FONT_SMALL)

    legend_x = left + 4 * cell_w + 70
    draw_text(draw, (legend_x, top + 4), "Composition", fnt=FONT_H3)
    for i, text in enumerate([
        "atom footprint: 16 M x 32 N",
        "atom depth: 64 K",
        "TiledMMA panel: 128 x 32 x 128",
        "full block: four N panels",
        "full block shape: 128 x 128 x 128",
    ]):
        draw_text(draw, (legend_x, top + 34 + i * 24), text, fill=(58, 65, 78))

    k_top = top + M_GROUPS * cell_h + 96
    draw_text(draw, (left, k_top - 34), "K composition inside each TiledMMA panel", fnt=FONT_H2)
    for phase in range(2):
        x0 = left + phase * 220
        color = (109, 127, 216) if phase == 0 else (193, 91, 115)
        rect(draw, (x0, k_top, x0 + 200, k_top + 70), color, (255, 255, 255), 2)
        draw_text(draw, (x0 + 58, k_top + 20), f"K phase {phase}", fill=(255, 255, 255), fnt=FONT_H3)
        draw_text(draw, (x0 + 52, k_top + 42), f"{phase * 64}-{phase * 64 + 63}", fill=(245, 248, 255), fnt=FONT_SMALL)
    draw_text(draw, (left + 460, k_top + 12), "Each 16x32 atom is evaluated at K0 and K1 to cover HeadDim=128.", fill=(58, 65, 78))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(TILER_PNG)


def render_thread_layout() -> None:
    img, draw = new_canvas(
        1160,
        660,
        "thread layout: warp groups and lane coordinates",
        "M-group = warp, thread = warp * 32 + lane. One warp expands as the MMA atom thread coordinate (_4,_8).",
    )

    left, top = 64, 128
    draw_text(draw, (left, top - 34), "8 warp overview", fnt=FONT_H2)
    lane_w, lane_h = 18, 22
    for lane in range(0, LANES, 4):
        draw_text(draw, (left + lane * lane_w - 2, top - 12), str(lane), fill=(77, 85, 99), fnt=FONT_SMALL)
    for warp in range(M_GROUPS):
        y0 = top + warp * lane_h
        draw_text(draw, (left - 48, y0 + 5), f"G{warp}", fill=(77, 85, 99), fnt=FONT_SMALL)
        for lane in range(LANES):
            thread = warp * LANES + lane
            x0 = left + lane * lane_w
            rect(draw, (x0, y0, x0 + lane_w, y0 + lane_h), thread_color(thread), (245, 247, 250), 1)
            if lane in (0, 8, 16, 24, 31):
                draw_text(draw, (x0 + 2, y0 + 6), str(lane), fill=(255, 255, 255), fnt=FONT_SMALL)
    draw_text(draw, (left + LANES * lane_w + 26, top + 30), "G0 = warp 0 = T0-T31\nG7 = warp 7 = T224-T255", fill=(58, 65, 78))

    zoom_x, zoom_y = 64, 380
    draw_text(draw, (zoom_x, zoom_y - 38), "Single-warp detail: G0 / T0-T31", fnt=FONT_H2)
    draw_text(draw, (zoom_x, zoom_y - 14), "MMA atom thread coordinate is (_4,_8): lane = row * 8 + col", fill=(82, 90, 105), fnt=FONT_SMALL)
    cell_w, cell_h = 88, 48
    for row in range(4):
        draw_text(draw, (zoom_x - 30, zoom_y + row * cell_h + 17), f"r{row}", fill=(77, 85, 99), fnt=FONT_SMALL)
        for col in range(8):
            lane = row * 8 + col
            x0 = zoom_x + col * cell_w
            y0 = zoom_y + row * cell_h
            color = thread_color(lane)
            rect(draw, (x0, y0, x0 + cell_w, y0 + cell_h), color, (255, 255, 255), 2)
            draw_text(draw, (x0 + 8, y0 + 9), f"lane {lane}", fill=contrast_color(color), fnt=FONT_SMALL)
            draw_text(draw, (x0 + 8, y0 + 26), f"T{lane}", fill=contrast_color(color), fnt=FONT_SMALL)
    for col in range(8):
        draw_text(draw, (zoom_x + col * cell_w + 33, zoom_y + 4 * cell_h + 8), f"c{col}", fill=(77, 85, 99), fnt=FONT_SMALL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(THREAD_PNG)


def lane_for_a(m: int, k: int) -> int:
    lane_col = m % 8
    lane_row = (k % 32) // 8
    return lane_row * 8 + lane_col


def lane_for_b(n: int, k: int) -> int:
    lane_col = n % 8
    lane_row = (k % 32) // 8
    return lane_row * 8 + lane_col


def lane_for_c(m: int, n: int) -> int:
    lane_col = m % 8
    lane_row = (n % 8) // 2
    return lane_row * 8 + lane_col


def render_fragment_partition(
    filename: Path,
    title: str,
    subtitle: str,
    rows: int,
    cols: int,
    row_label: str,
    col_label: str,
    lane_fn,
    formula: list[str],
    example: list[str],
    cell_w: int = 18,
    cell_h: int = 22,
    major_x: int = 8,
    major_y: int = 8,
) -> None:
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    width = max(1220, grid_w + 360)
    height = grid_h + 250
    img, draw = new_canvas(width, height, title, subtitle)

    x0, y0 = 64, 142
    draw_text(draw, (x0, y0 - 30), f"{row_label} x {col_label} ownership by lane", fnt=FONT_H2)
    for col in range(0, cols, 2 if cols <= 32 else 4):
        draw_text(draw, (x0 + col * cell_w + 2, y0 - 18), str(col), fill=(77, 85, 99), fnt=FONT_TINY)
    for row in range(rows):
        draw_text(draw, (x0 - 28, y0 + row * cell_h + 5), str(row), fill=(77, 85, 99), fnt=FONT_TINY)
        for col in range(cols):
            lane = lane_fn(row, col)
            color = thread_color(lane)
            cx = x0 + col * cell_w
            cy = y0 + row * cell_h
            rect(draw, (cx, cy, cx + cell_w, cy + cell_h), color, (255, 255, 255), 1)
            draw_text(draw, (cx + 4, cy + 5), str(lane), fill=contrast_color(color), fnt=FONT_TINY)

    draw.rectangle((x0, y0, x0 + grid_w, y0 + grid_h), outline=(65, 72, 84), width=2)
    for col in range(0, cols + 1, major_x):
        x = x0 + col * cell_w
        draw.line((x, y0, x, y0 + grid_h), fill=(20, 24, 32), width=2)
    for row in range(0, rows + 1, major_y):
        y = y0 + row * cell_h
        draw.line((x0, y, x0 + grid_w, y), fill=(20, 24, 32), width=2)

    info_x = x0 + grid_w + 48
    draw_text(draw, (info_x, y0 + 4), "Mapping", fnt=FONT_H3)
    for i, text in enumerate(formula):
        draw_text(draw, (info_x, y0 + 34 + i * 24), text, fill=(58, 65, 78))
    ex_y = y0 + 34 + len(formula) * 24 + 34
    draw_text(draw, (info_x, ex_y), "Example: lane 0 / T0", fnt=FONT_H3)
    for i, text in enumerate(example):
        draw_text(draw, (info_x, ex_y + 30 + i * 24), text, fill=(58, 65, 78))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(filename)


def render_fragment_partitions() -> None:
    render_fragment_partition(
        FRAG_A_PNG,
        "AtomLayoutA_TV fragment A partition",
        "LayoutA_TV = ((_4,_8),(_8,_2,_2)):((_128,_1),(_16,_8,_512)); interpreted as A(M,K)=16x64.",
        rows=16,
        cols=64,
        row_label="M",
        col_label="K",
        lane_fn=lane_for_a,
        formula=[
            "lane row r = lane / 8",
            "lane col c = lane % 8",
            "m = c or c + 8",
            "k = 8*r + {0..7} + 32*p",
            "p in {0,1}",
        ],
        example=[
            "m = 0 and 8",
            "k = 0..7 and 32..39",
            "32 A values total",
        ],
        cell_w=16,
        cell_h=22,
        major_x=8,
        major_y=8,
    )
    render_fragment_partition(
        FRAG_B_PNG,
        "AtomLayoutB_TV fragment B partition",
        "LayoutB_TV = ((_4,_8),(_8,_2,_4)):((_256,_1),(_32,_1024,_8)); interpreted as B(N,K)=32x64.",
        rows=32,
        cols=64,
        row_label="N",
        col_label="K",
        lane_fn=lane_for_b,
        formula=[
            "lane row r = lane / 8",
            "lane col c = lane % 8",
            "n = c + 8*q",
            "k = 8*r + {0..7} + 32*p",
            "q in {0..3}, p in {0,1}",
        ],
        example=[
            "n = 0, 8, 16, 24",
            "k = 0..7 and 32..39",
            "64 B values total",
        ],
        cell_w=16,
        cell_h=16,
        major_x=8,
        major_y=8,
    )
    render_fragment_partition(
        FRAG_C_PNG,
        "AtomLayoutC_TV accumulator C partition",
        "LayoutC_TV = ((_4,_8),((_2,_4),_2)):((_32,_1),((_16,_128),_8)); interpreted as C(M,N)=16x32.",
        rows=16,
        cols=32,
        row_label="M",
        col_label="N",
        lane_fn=lane_for_c,
        formula=[
            "lane row r = lane / 8",
            "lane col c = lane % 8",
            "m = c or c + 8",
            "n = 2*r + {0,1} + 8*p",
            "p in {0..3}",
        ],
        example=[
            "m = 0 and 8",
            "n = 0,1, 8,9, 16,17, 24,25",
            "16 accumulator values total",
        ],
        cell_w=27,
        cell_h=21,
        major_x=8,
        major_y=8,
    )


def find_debug_binary(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path("/tmp/sageattn4_blackwell_cmake_debug/mainloop_tma_ws_debug"),
        THIS_DIR / "cmake-build-debug" / "mainloop_tma_ws_debug",
        THIS_DIR / "build" / "mainloop_tma_ws_debug",
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise SystemExit(
        "Could not find mainloop_tma_ws_debug. Build it first or pass "
        "--debug-bin /path/to/mainloop_tma_ws_debug."
    )


def new_owner_map() -> list[list[int]]:
    return [[-1 for _ in range(TILE_N)] for _ in range(TILE_M)]


def set_owner(owner: list[list[int]], m: int, n: int, thread: int) -> None:
    if not (0 <= m < TILE_M and 0 <= n < TILE_N):
        return
    old = owner[m][n]
    if old == -1 or old == thread:
        owner[m][n] = thread
    else:
        owner[m][n] = -2


def load_access_maps(debug_bin: Path) -> tuple[list[list[int]], dict[str, list[list[list[list[int]]]]]]:
    q_owner = new_owner_map()
    staged = {
        "SmemLayoutK": [[new_owner_map() for _ in range(STAGES)] for _ in range(M_GROUPS)],
        "SmemLayoutVt": [[new_owner_map() for _ in range(STAGES)] for _ in range(M_GROUPS)],
    }

    proc = subprocess.Popen(
        [str(debug_bin), "--access-map-csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    header_seen = False
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        if line == "layout,thread,value,m,n_or_k,stage":
            header_seen = True
            continue
        if not header_seen:
            continue
        layout, thread_s, _value_s, m_s, n_s, stage_s = line.split(",")
        thread = int(thread_s)
        m = int(m_s)
        n = int(n_s)
        stage = int(stage_s)
        if layout == "SmemLayoutQ":
            set_owner(q_owner, m, n, thread)
        elif layout in staged:
            group = thread // LANES
            if 0 <= group < M_GROUPS and 0 <= stage < STAGES:
                set_owner(staged[layout][group][stage], m, n, thread)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"{debug_bin} --access-map-csv failed with {rc}:\n{stderr}")
    if not header_seen:
        raise SystemExit(f"{debug_bin} did not emit access-map CSV")
    return q_owner, staged


def map_to_image(owner: list[list[int]], scale: int) -> Image.Image:
    small = Image.new("RGB", (TILE_N, TILE_M), (230, 234, 240))
    px = small.load()
    for m in range(TILE_M):
        for n in range(TILE_N):
            thread = owner[m][n]
            if thread >= 0:
                px[n, m] = thread_color(thread)
            elif thread == -2:
                px[n, m] = (20, 24, 32)
    return small.resize((TILE_N * scale, TILE_M * scale), Image.Resampling.NEAREST)


def draw_grid_overlay(draw: ImageDraw.ImageDraw, x: int, y: int, scale: int) -> None:
    size = TILE_M * scale
    for i in range(0, TILE_M + 1, 16):
        p = x + i * scale
        draw.line((p, y, p, y + size), fill=(255, 255, 255), width=1)
        p = y + i * scale
        draw.line((x, p, x + size, p), fill=(255, 255, 255), width=1)
    draw.rectangle((x, y, x + size, y + size), outline=(65, 72, 84), width=2)


def draw_thread_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw_text(draw, (x, y), "Thread color samples", fnt=FONT_H3)
    samples = [0, 1, 31, 32, 63, 64, 127, 128, 191, 224, 255]
    for i, thread in enumerate(samples):
        sx = x + (i % 6) * 84
        sy = y + 28 + (i // 6) * 26
        rect(draw, (sx, sy, sx + 18, sy + 18), thread_color(thread), (255, 255, 255), 1)
        draw_text(draw, (sx + 24, sy + 2), f"T{thread}", fnt=FONT_SMALL)
    draw_text(draw, (x, y + 86), "Black means multiple thread IDs touched the same coordinate in that panel.", fill=(74, 83, 97), fnt=FONT_SMALL)


def render_smem_q(q_owner: list[list[int]]) -> None:
    img, draw = new_canvas(
        1180,
        820,
        "SmemLayoutQ thread access",
        "CuTe identity tensor partitioned by make_tiled_copy_A. Color is thread ID.",
    )
    x, y, scale = 48, 138, 5
    draw_text(draw, (x, y - 28), "128 x 128 Q tile, T0-T255", fnt=FONT_H2)
    img.paste(map_to_image(q_owner, scale), (x, y))
    draw_grid_overlay(draw, x, y, scale)
    draw_thread_legend(draw, 760, 160)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(SMEM_Q_PNG)


def render_smem_staged(filename: Path, title: str, staged_maps: list[list[list[list[int]]]]) -> None:
    scale = 2
    panel = TILE_M * scale
    gap_x, gap_y = 22, 34
    img, draw = new_canvas(
        1080,
        2840,
        title,
        "CuTe identity tensor partitioned by make_tiled_copy_B. Split by warp/M-group and pipeline stage.",
    )
    x, y = 48, 138
    for stage in range(STAGES):
        draw_text(draw, (x + 86 + stage * (panel + gap_x), y - 28), f"stage {stage}", fnt=FONT_H3)
    for group in range(M_GROUPS):
        row_y = y + group * (panel + gap_y)
        draw_text(draw, (x, row_y + 98), f"G{group}\nT{group * 32}-T{group * 32 + 31}", fill=(58, 65, 78), fnt=FONT_SMALL)
        for stage in range(STAGES):
            px = x + 86 + stage * (panel + gap_x)
            img.paste(map_to_image(staged_maps[group][stage], scale), (px, row_y))
            draw_grid_overlay(draw, px, row_y, scale)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(filename)


def render_shared_access(debug_bin: Path) -> None:
    q_owner, staged = load_access_maps(debug_bin)
    render_smem_q(q_owner)
    render_smem_staged(SMEM_K_PNG, "SmemLayoutK thread access", staged["SmemLayoutK"])
    render_smem_staged(SMEM_VT_PNG, "SmemLayoutVt thread access", staged["SmemLayoutVt"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-bin", help="Path to the built mainloop_tma_ws_debug executable")
    args = parser.parse_args()

    debug_bin = find_debug_binary(args.debug_bin)
    render_tiler_composition()
    render_thread_layout()
    render_fragment_partitions()
    render_shared_access(debug_bin)
    for output in [
        TILER_PNG,
        THREAD_PNG,
        FRAG_A_PNG,
        FRAG_B_PNG,
        FRAG_C_PNG,
        SMEM_Q_PNG,
        SMEM_K_PNG,
        SMEM_VT_PNG,
    ]:
        print(output)


if __name__ == "__main__":
    main()
