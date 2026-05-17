#!/usr/bin/env python3
"""Render PNG diagrams for explicit layouts in the generated blockmean1 header.

The C++ header is the source of truth. This script parses every
``using *Layout* = decltype(make_layout(...))`` or composed swizzle layout,
uses CuTe DSL to evaluate static sample views, and writes compact heatmaps
showing the resulting address/index mapping.
"""

from __future__ import annotations

import argparse
import colorsys
import itertools
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_HEADER = THIS_DIR / "sageattn4_fwd_hdim128_bm128_bn128_s3_blockmean1.h"
DEFAULT_OUTPUT_DIR = THIS_DIR / "layout_visualizations"


@dataclass(frozen=True)
class DynamicValue:
    kind: str
    ordinal: int


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    shape: object
    stride: object
    source_shape: str
    source_stride: str
    swizzle: tuple[int, int, int] | None = None
    smem_flag_bits: int | None = None


@dataclass(frozen=True)
class LayoutView:
    spec: LayoutSpec
    suffix: str
    coords: tuple[int, ...]
    rows: int
    cols: int

    @property
    def name(self) -> str:
        if not self.suffix:
            return self.spec.name
        return f"{self.spec.name}_{self.suffix}"


@dataclass
class EvaluatedView:
    view: LayoutView
    values: list[list[int]]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = _font(24, True)
FONT = _font(13)
FONT_SMALL = _font(11)
FONT_TINY = _font(8)


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    angle = 0
    for i, ch in enumerate(text):
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle -= 1
        elif angle == 0:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(text[start:i].strip())
                start = i + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def find_matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unmatched '(' at byte {open_pos}")


def call_inner(text: str, function: str, start: int = 0) -> tuple[str, int, int]:
    needle = f"{function}("
    pos = text.find(needle, start)
    if pos < 0:
        raise ValueError(f"could not find {function}(...)")
    open_pos = pos + len(function)
    close_pos = find_matching_paren(text, open_pos)
    return text[open_pos + 1:close_pos], pos, close_pos


def parse_int_tuple(expr: str, dynamic_kind: str, counter: list[int]) -> object:
    expr = expr.strip()
    if expr.startswith("make_shape("):
        inner, _, _ = call_inner(expr, "make_shape")
        return tuple(parse_int_tuple(part, dynamic_kind, counter) for part in split_top_level(inner))
    if expr.startswith("make_stride("):
        inner, _, _ = call_inner(expr, "make_stride")
        return tuple(parse_int_tuple(part, dynamic_kind, counter) for part in split_top_level(inner))
    match = re.fullmatch(r"Int<\s*(-?\d+)\s*>\s*\{\}", expr)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"int32_t\s*\{\}", expr):
        ordinal = counter[0]
        counter[0] += 1
        return DynamicValue(dynamic_kind, ordinal)
    raise ValueError(f"unsupported tuple expression: {expr}")


def parse_make_layout(body: str) -> tuple[object, object, str, str]:
    inner, _, _ = call_inner(body, "make_layout")
    args = split_top_level(inner)
    if len(args) != 2:
        raise ValueError(f"expected make_layout(shape, stride), got {len(args)} args")
    shape_counter = [0]
    stride_counter = [0]
    shape = parse_int_tuple(args[0], "shape", shape_counter)
    stride = parse_int_tuple(args[1], "stride", stride_counter)
    return shape, stride, args[0], args[1]


def parse_layout_specs(header: Path) -> list[LayoutSpec]:
    text = header.read_text()
    specs: list[LayoutSpec] = []
    pattern = re.compile(r"\busing\s+(\w*Layout\w*)\s*=\s*decltype\s*\(")
    for match in pattern.finditer(text):
        name = match.group(1)
        open_pos = text.find("(", match.end() - 1)
        close_pos = find_matching_paren(text, open_pos)
        body = text[open_pos + 1:close_pos]
        shape, stride, source_shape, source_stride = parse_make_layout(body)
        swizzle_match = re.search(r"Swizzle<\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*>\s*\{\}", body)
        flag_match = re.search(r"smem_ptr_flag_bits<\s*(\d+)\s*>\s*\{\}", body)
        specs.append(
            LayoutSpec(
                name=name,
                shape=shape,
                stride=stride,
                source_shape=source_shape,
                source_stride=source_stride,
                swizzle=tuple(int(swizzle_match.group(i)) for i in range(1, 4)) if swizzle_match else None,
                smem_flag_bits=int(flag_match.group(1)) if flag_match else None,
            )
        )
    specs.extend(parse_known_external_atom_specs(text, {spec.name for spec in specs}))
    return specs


def parse_known_external_atom_specs(text: str, existing_names: set[str]) -> list[LayoutSpec]:
    specs: list[LayoutSpec] = []
    pattern = re.compile(
        r"\busing\s+(\w*Layout\w*)\s*=\s*"
        r"(?:UMMA|GMMA)::Layout_K_SW(64|128)_Atom<\s*([^>]+?)\s*>\s*;"
    )
    element_bits = {
        "cutlass::float_e2m1_t": 4,
        "cutlass::bfloat16_t": 16,
    }
    for match in pattern.finditer(text):
        name, swizzle_width, element_type = match.groups()
        if name in existing_names:
            continue
        bits = element_bits.get(element_type)
        if bits is None:
            continue
        width = int(swizzle_width)
        base_cols = 512 if width == 64 else 1024
        base_swizzle = (2, 4, 3) if width == 64 else (3, 4, 3)
        log2_bits = bits.bit_length() - 1
        swizzle_m = base_swizzle[1] - log2_bits
        if swizzle_m < 0:
            swizzle = (max(base_swizzle[0] + swizzle_m, 0), 0, base_swizzle[2])
        else:
            swizzle = (base_swizzle[0], swizzle_m, base_swizzle[2])
        shape = (8, base_cols // bits)
        stride = (base_cols // bits, 1)
        specs.append(
            LayoutSpec(
                name=name,
                shape=shape,
                stride=stride,
                source_shape=f"CUTLASS Layout_K_SW{width}_Atom<{element_type}> upcast shape {shape}",
                source_stride=f"CUTLASS Layout_K_SW{width}_Atom<{element_type}> upcast stride {stride}",
                swizzle=swizzle,
                smem_flag_bits=bits,
            )
        )
    return specs


def substitute_dynamic(value: object, *, dynamic_shape: int, dynamic_stride: int) -> object:
    if isinstance(value, DynamicValue):
        return dynamic_shape if value.kind == "shape" else dynamic_stride
    if isinstance(value, tuple):
        return tuple(substitute_dynamic(item, dynamic_shape=dynamic_shape, dynamic_stride=dynamic_stride) for item in value)
    return value


def product(value: object) -> int:
    if isinstance(value, tuple):
        result = 1
        for item in value:
            result *= product(item)
        return result
    return int(value)


def tuple_rank(shape: object) -> int:
    return len(shape) if isinstance(shape, tuple) else 1


def mode(shape: object, index: int) -> object:
    if isinstance(shape, tuple):
        return shape[index]
    if index == 0:
        return shape
    raise IndexError(index)


def has_dynamic(value: object) -> bool:
    if isinstance(value, DynamicValue):
        return True
    if isinstance(value, tuple):
        return any(has_dynamic(item) for item in value)
    return False


def make_views(specs: Iterable[LayoutSpec], *, dynamic_shape: int, dynamic_stride: int) -> list[LayoutView]:
    views: list[LayoutView] = []
    for raw_spec in specs:
        shape = substitute_dynamic(raw_spec.shape, dynamic_shape=dynamic_shape, dynamic_stride=dynamic_stride)
        stride = substitute_dynamic(raw_spec.stride, dynamic_shape=dynamic_shape, dynamic_stride=dynamic_stride)
        spec = LayoutSpec(
            name=raw_spec.name,
            shape=shape,
            stride=stride,
            source_shape=raw_spec.source_shape,
            source_stride=raw_spec.source_stride,
            swizzle=raw_spec.swizzle,
            smem_flag_bits=raw_spec.smem_flag_bits,
        )
        rank = tuple_rank(shape)
        rows = product(mode(shape, 0))
        cols = product(mode(shape, 1)) if rank >= 2 else 1
        if rank <= 2:
            views.append(LayoutView(spec, "", (), rows, cols))
            continue

        extra_sizes = [product(mode(shape, i)) for i in range(2, rank)]
        first_extra = extra_sizes[0]
        if first_extra <= 4:
            slice_coords = [(i, *([0] * (len(extra_sizes) - 1))) for i in range(first_extra)]
        else:
            slice_coords = [tuple(0 for _ in extra_sizes)]
        for coords in slice_coords:
            suffix = "_".join(f"mode{i + 2}_{coord}" for i, coord in enumerate(coords))
            views.append(LayoutView(spec, suffix, tuple(coords), rows, cols))
    return views


def py_literal(value: object) -> str:
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({py_literal(value[0])},)"
        return "(" + ", ".join(py_literal(item) for item in value) + ")"
    return str(int(value))


def make_dsl_probe(views: list[LayoutView]) -> str:
    chunks = [
        "import cutlass",
        "import cutlass.cute as cute",
        "",
    ]
    calls: list[str] = []
    for i, view in enumerate(views):
        spec = view.spec
        shape = py_literal(spec.shape)
        stride = py_literal(spec.stride)
        coords_tail = ", ".join(str(coord) for coord in view.coords)
        if coords_tail:
            coord_expr = f"(m, n, {coords_tail})" if tuple_rank(spec.shape) >= 2 else f"(m, {coords_tail})"
        else:
            coord_expr = "(m, n)" if tuple_rank(spec.shape) >= 2 else "(m,)"
        chunks.append("@cute.jit")
        chunks.append(f"def emit_{i}():")
        chunks.append(f"    inner = cute.make_layout({shape}, stride={stride})")
        if spec.swizzle:
            b, m_bits, s = spec.swizzle
            chunks.append(f"    layout = cute.make_composed_layout(cute.make_swizzle({b}, {m_bits}, {s}), 0, inner)")
        else:
            chunks.append("    layout = inner")
        chunks.append(f"    print('BEGIN_VIEW {i} {view.rows} {view.cols}')")
        chunks.append(f"    for m in cutlass.range_constexpr({view.rows}):")
        chunks.append(f"        for n in cutlass.range_constexpr({view.cols}):")
        chunks.append(f"            idx = layout({coord_expr})")
        chunks.append("            print('%d,%d,%d' % (m, n, idx))")
        chunks.append(f"    print('END_VIEW {i}')")
        chunks.append("")
        calls.append(f"emit_{i}()")
    chunks.append("if __name__ == '__main__':")
    chunks.extend(f"    {call}" for call in calls)
    return "\n".join(chunks) + "\n"


def evaluate_views_with_cutedsl(views: list[LayoutView]) -> list[EvaluatedView]:
    if not views:
        return []
    with tempfile.TemporaryDirectory(prefix="sageattn4_layouts_") as tmp_dir:
        probe_path = Path(tmp_dir) / "probe_layouts.py"
        probe_path.write_text(make_dsl_probe(views))
        proc = subprocess.run(
            [sys.executable, str(probe_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        raise SystemExit(f"CuTe DSL layout probe failed with {proc.returncode}:\n{proc.stderr}")

    matrices: dict[int, list[list[int]]] = {}
    current: int | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("BEGIN_VIEW "):
            _, idx_s, rows_s, cols_s = line.split()
            idx = int(idx_s)
            matrices[idx] = [[0 for _ in range(int(cols_s))] for _ in range(int(rows_s))]
            current = idx
            continue
        if line.startswith("END_VIEW "):
            current = None
            continue
        if current is None or not line or "," not in line:
            continue
        row_s, col_s, value_s = line.split(",", 2)
        matrices[current][int(row_s)][int(col_s)] = int(value_s)

    missing = [i for i in range(len(views)) if i not in matrices]
    if missing:
        raise SystemExit(f"CuTe DSL probe did not emit views: {missing}")
    return [EvaluatedView(views[i], matrices[i]) for i in range(len(views))]


def color_for_value(value: int) -> tuple[int, int, int]:
    hue = ((value * 0.61803398875) % 1.0)
    sat = 0.42 + 0.16 * ((value >> 4) & 1)
    val = 0.88 - 0.14 * ((value >> 5) & 1)
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return tuple(int(c * 255) for c in rgb)


def contrast(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 24, 32) if lum > 150 else (255, 255, 255)


def scale_for(rows: int, cols: int) -> int:
    largest = max(rows, cols)
    if largest <= 8:
        return 42
    if largest <= 16:
        return 30
    if largest <= 32:
        return 18
    if largest <= 64:
        return 10
    return 5


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 4,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + line_gap
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, fill=fill, font=font)
    return y + max(1, len(lines)) * line_h


def image_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).lower() + ".png"


def render_view(evaluated: EvaluatedView, output_dir: Path) -> Path:
    view = evaluated.view
    rows, cols = view.rows, view.cols
    scale = scale_for(rows, cols)
    grid_w = cols * scale
    grid_h = rows * scale
    pad_x = 36
    top = 132
    side_w = 360
    width = max(760, pad_x * 2 + grid_w + side_w)
    height = max(620, top + grid_h + 48)

    img = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(img)
    draw.text((28, 22), view.name, fill=(24, 30, 41), font=FONT_TITLE)

    subtitle = f"{rows} x {cols} view"
    if view.coords:
        subtitle += f"; fixed extra coords={view.coords}"
    if view.spec.swizzle:
        subtitle += f"; Swizzle{view.spec.swizzle}"
    draw.text((28, 56), subtitle, fill=(82, 90, 105), font=FONT)

    flat_values = list(itertools.chain.from_iterable(evaluated.values))
    unique_values = len(set(flat_values))
    min_value = min(flat_values)
    max_value = max(flat_values)
    dynamic_note = ""
    if "int32_t" in view.spec.source_shape or "int32_t" in view.spec.source_stride:
        dynamic_note = "Dynamic int32_t{} slots use sampled static values from script arguments."

    x0, y0 = pad_x, top
    pixels = Image.new("RGB", (cols, rows), (228, 233, 240))
    px = pixels.load()
    for r in range(rows):
        for c in range(cols):
            px[c, r] = color_for_value(evaluated.values[r][c])
    img.paste(pixels.resize((grid_w, grid_h), Image.Resampling.NEAREST), (x0, y0))

    for c in range(cols + 1):
        if cols <= 32 or c % 16 == 0:
            x = x0 + c * scale
            draw.line((x, y0, x, y0 + grid_h), fill=(255, 255, 255), width=1)
    for r in range(rows + 1):
        if rows <= 32 or r % 16 == 0:
            y = y0 + r * scale
            draw.line((x0, y, x0 + grid_w, y), fill=(255, 255, 255), width=1)
    draw.rectangle((x0, y0, x0 + grid_w, y0 + grid_h), outline=(48, 56, 68), width=2)

    if rows <= 16 and cols <= 16:
        for r in range(rows):
            for c in range(cols):
                value = evaluated.values[r][c]
                fill = contrast(color_for_value(value))
                text = str(value)
                bbox = draw.textbbox((0, 0), text, font=FONT_TINY)
                tx = x0 + c * scale + (scale - (bbox[2] - bbox[0])) / 2
                ty = y0 + r * scale + (scale - (bbox[3] - bbox[1])) / 2 - 1
                draw.text((tx, ty), text, fill=fill, font=FONT_TINY)

    info_x = x0 + grid_w + 34
    y = top + 4
    draw.text((info_x, y), "Mapping", fill=(24, 30, 41), font=FONT_TITLE)
    y += 42
    for line in [
        f"min index: {min_value}",
        f"max index: {max_value}",
        f"unique indices: {unique_values}",
        f"collisions: {rows * cols - unique_values}",
    ]:
        draw.text((info_x, y), line, fill=(61, 69, 82), font=FONT)
        y += 23
    y += 8
    shape_text = f"shape: {view.spec.source_shape}"
    stride_text = f"stride: {view.spec.source_stride}"
    y = draw_wrapped_text(draw, (info_x, y), shape_text, width=side_w - 40, font=FONT_SMALL, fill=(61, 69, 82))
    y += 8
    y = draw_wrapped_text(draw, (info_x, y), stride_text, width=side_w - 40, font=FONT_SMALL, fill=(61, 69, 82))
    if dynamic_note:
        y += 12
        draw_wrapped_text(draw, (info_x, y), dynamic_note, width=side_w - 40, font=FONT_SMALL, fill=(126, 80, 32))
    if view.spec.smem_flag_bits is not None:
        y += 12
        draw_wrapped_text(
            draw,
            (info_x, y),
            f"smem_ptr_flag_bits<{view.spec.smem_flag_bits}> is modeled as zero offset for relative indexing.",
            width=side_w - 40,
            font=FONT_SMALL,
            fill=(126, 80, 32),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / image_name(view.name)
    img.save(path)
    return path


def write_index(
    output_dir: Path,
    header: Path,
    raw_specs: list[LayoutSpec],
    evaluated: list[EvaluatedView],
    paths: list[Path],
    *,
    dynamic_shape: int,
    dynamic_stride: int,
) -> None:
    explicit_names = {spec.name for spec in raw_specs}
    header_text = header.read_text()
    listed_names = re.findall(r"\busing\s+(\w*Layout\w*)\s*=", header_text)
    external = [name for name in listed_names if name not in explicit_names]

    lines = [
        "# SageAttention4 Blockmean1 Layout Visualizations",
        "",
        f"Header: `{header.name}`",
        f"CuTe DSL evaluated {len(evaluated)} rendered views from {len(raw_specs)} layout aliases with inline or known CUTLASS atom definitions.",
        f"Dynamic `int32_t{{}}` sample values: shape={dynamic_shape}, stride={dynamic_stride}.",
        "",
        "Each PNG is a 2D view of the CuTe layout index mapping. Rank-3+ layouts are sliced over the first extra mode when that mode is small.",
        "",
    ]
    if external:
        lines.extend(
            [
                "Layout aliases without inline `make_layout(...)` definitions or a known CUTLASS atom pattern are not rendered directly:",
                "",
            ]
        )
        lines.extend(f"- `{name}`" for name in external)
        lines.append("")

    lines.append("## Rendered Views")
    lines.append("")
    for view, path in zip((item.view for item in evaluated), paths):
        rel = path.relative_to(output_dir)
        lines.append(f"- [{view.name}]({rel.as_posix()})")
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamic-shape", type=int, default=2, help="sample extent for int32_t{} shape slots")
    parser.add_argument("--dynamic-stride", type=int, default=1024, help="sample stride for int32_t{} stride slots")
    args = parser.parse_args()

    specs = parse_layout_specs(args.header)
    if not specs:
        raise SystemExit(f"No explicit layout aliases found in {args.header}")
    views = make_views(specs, dynamic_shape=args.dynamic_shape, dynamic_stride=args.dynamic_stride)
    evaluated = evaluate_views_with_cutedsl(views)
    paths = [render_view(item, args.output_dir) for item in evaluated]
    write_index(
        args.output_dir,
        args.header,
        specs,
        evaluated,
        paths,
        dynamic_shape=args.dynamic_shape,
        dynamic_stride=args.dynamic_stride,
    )
    print(f"Rendered {len(paths)} layout views into {args.output_dir}")


if __name__ == "__main__":
    main()
