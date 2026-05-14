"""C++ type synthesis helpers for SageAttention4 Blackwell generators."""

from __future__ import annotations

import ast
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

try:
    import cutlass as _cutlass
    import cutlass.cute as _cute
    from cutlass._mlir import ir as _ir

    CUTEDSL_STATUS = "available"
except Exception:
    _cutlass = None
    _cute = None
    _ir = None
    CUTEDSL_STATUS = "unavailable"


IntTuple = int | tuple["IntTuple", ...]
CPP_DTYPE_NAMES = {
    "BFloat16": "cutlass::bfloat16_t",
    "Float4E2M1FN": "cutlass::float_e2m1_t",
    "Float8E4M3FN": "cutlass::float_ue4m3_t",
    "Float32": "float",
    "Int64": "int64_t",
}
DTYPE_BITS = {
    "BFloat16": 16,
    "Float4E2M1FN": 4,
    "Float8E4M3FN": 8,
    "Float32": 32,
    "Int64": 64,
}
IR_DTYPE_NAMES = {
    "bf16": "cutlass::bfloat16_t",
    "f4E2M1FN": "cutlass::float_e2m1_t",
    "f8E4M3FN": "cutlass::float_ue4m3_t",
    "f32": "float",
    "i64": "int64_t",
}
_CUTEDSL_KEEPALIVE: list[Any] = []


class CppType:
    def to_cpp(self) -> str:
        raise NotImplementedError

    def instance(self) -> str:
        return f"{self.to_cpp()}{{}}"


class CppExpr:
    def to_cpp(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedType(CppType):
    name: str

    def to_cpp(self) -> str:
        return self.name


@dataclass(frozen=True)
class DynamicInt(CppType):
    def to_cpp(self) -> str:
        return "int32_t"


LayoutTuple = int | DynamicInt | tuple["LayoutTuple", ...]


@dataclass(frozen=True)
class CuteInt(CppType):
    value: int

    def to_cpp(self) -> str:
        return f"cute::Int<{self.value}>"


@dataclass(frozen=True)
class CutlassDType(CppType):
    value: object

    def to_cpp(self) -> str:
        return render_cutlass_dtype(self.value)


@dataclass(frozen=True)
class CuteTuple(CppType):
    value: Any
    wrapper: str

    def to_cpp(self) -> str:
        return render_cute_tuple(self.value, self.wrapper)


@dataclass(frozen=True)
class CuteLayout(CppType):
    shape: IntTuple
    stride: IntTuple | None = None
    dsl_value: Any = field(init=False, repr=False, compare=False)
    dsl_shape: Any = field(init=False, repr=False, compare=False)
    dsl_stride: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dsl_layout = make_cutedsl_layout(self.shape, self.stride)
        object.__setattr__(self, "dsl_value", dsl_layout)
        object.__setattr__(self, "dsl_shape", tupleify(dsl_layout.shape))
        object.__setattr__(self, "dsl_stride", tupleify(dsl_layout.stride))

    def to_cpp(self) -> str:
        return render_layout(self.dsl_shape, self.dsl_stride)


@dataclass(frozen=True)
class CuteConcreteLayout(CppType):
    dsl_value: Any = field(repr=False, compare=False)
    dsl_shape: Any
    dsl_stride: Any

    def to_cpp(self) -> str:
        return render_layout(self.dsl_shape, self.dsl_stride)


@dataclass(frozen=True)
class CuteComposedLayout(CppType):
    dsl_value: Any = field(repr=False, compare=False)
    swizzle: tuple[int, int, int]
    offset: int | str | CppType
    outer_shape: Any
    outer_stride: Any

    def to_cpp(self) -> str:
        return render_composed_layout(self.swizzle, self.offset, self.outer_shape, self.outer_stride)


@dataclass(frozen=True)
class CuteSmemLayoutAtom(CppType):
    family: str
    major: str
    suffix: str
    element: CutlassDType
    swizzle: tuple[int, int, int]
    offset: int
    flag_bits: int
    dsl_value: Any = field(repr=False, compare=False)

    def to_cpp(self) -> str:
        return f"cute::{self.family}::Layout_{self.major}_{self.suffix}_Atom<{self.element.to_cpp()}>"


@dataclass(frozen=True)
class TemplateType(CppType):
    name: str
    args: tuple[str | CppType, ...]

    def to_cpp(self) -> str:
        rendered = [render_cpp(arg) for arg in self.args]
        return f"{self.name}<{', '.join(rendered)}>"


@dataclass(frozen=True)
class CuteTiledMma(CppType):
    ir_type: str

    def to_cpp(self) -> str:
        return render_tiled_mma_ir(self.ir_type)


@dataclass(frozen=True)
class CuteCopyAtom(CppType):
    ir_type: str

    def to_cpp(self) -> str:
        return render_copy_atom_ir(self.ir_type)


@dataclass(frozen=True)
class InstanceExpr(CppExpr):
    value: str | CppType

    def to_cpp(self) -> str:
        return f"{render_cpp(self.value)}{{}}"


@dataclass(frozen=True)
class CallExpr(CppExpr):
    name: str | CppType
    args: tuple[str | CppType | CppExpr, ...]
    multiline: bool = False

    def to_cpp(self) -> str:
        rendered = [render_cpp(arg) for arg in self.args]
        if not self.multiline:
            return f"{render_cpp(self.name)}(" + ", ".join(rendered) + ")"
        joined = ",\n    ".join(rendered)
        return f"{render_cpp(self.name)}(\n    {joined}\n)"


@dataclass(frozen=True)
class DecltypeType(CppType):
    expr: str | CppExpr

    def to_cpp(self) -> str:
        return f"decltype({render_cpp(self.expr)})"


def cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def named(cpp: str) -> NamedType:
    return NamedType(cpp)


def cute_int(value: int) -> CuteInt:
    return CuteInt(value)


def cutlass_dtype(name: str) -> CutlassDType:
    if _cutlass is None:
        raise RuntimeError("CUTLASS DSL is required to synthesize primitive dtype aliases")
    return CutlassDType(_cutlass.dtype(name))


def cute_shape(value: Any) -> CuteTuple:
    return CuteTuple(value, "Shape")


def cute_stride(value: Any) -> CuteTuple:
    return CuteTuple(value, "Stride")


def cute_layout(shape: IntTuple, *, stride: IntTuple | None = None) -> CuteLayout:
    return CuteLayout(shape, stride)


def template_type(name: str, *args: str | CppType) -> TemplateType:
    return TemplateType(name, args)


def dynamic_int() -> DynamicInt:
    return DynamicInt()


def universal_copy(element: str | CppType) -> TemplateType:
    return template_type("cute::UniversalCopy", element)


def sm120_rr_smem_selector(element: CutlassDType, major_size: int) -> CuteSmemLayoutAtom:
    return k_major_smem_selector("UMMA", element, major_size)


def gmma_k_smem_selector(element: CutlassDType, major_size: int) -> CuteSmemLayoutAtom:
    return k_major_smem_selector("GMMA", element, major_size)


def k_major_smem_selector(family: str, element: CutlassDType, major_size: int) -> CuteSmemLayoutAtom:
    element_bits = DTYPE_BITS[cutlass_dtype_name(element.value)]
    if family == "UMMA" and element_bits > 8:
        raise ValueError(f"unsupported SM120 RR shared-memory element size: {element_bits} bits")

    for swizzle_bits, suffix, swizzle in [
        (1024, "SW128", (3, 4, 3)),
        (512, "SW64", (2, 4, 3)),
        (256, "SW32", (1, 4, 3)),
        (128, "INTER", (0, 4, 3)),
    ]:
        atom_major_size = swizzle_bits // element_bits
        if major_size % atom_major_size == 0:
            dsl_value = make_cutedsl_composed_layout(
                swizzle,
                0,
                (8, atom_major_size),
                (atom_major_size, 1),
            )
            return CuteSmemLayoutAtom(family, "K", suffix, element, swizzle, 0, element_bits, dsl_value)

    raise ValueError(
        "no SM120 RR shared-memory layout atom for "
        f"{cutlass_dtype_name(element.value)} with major_size={major_size}"
    )


def decltype(expr: str | CppExpr) -> DecltypeType:
    return DecltypeType(expr)


def instance(value: str | CppType) -> InstanceExpr:
    return InstanceExpr(value)


def call_expr(name: str | CppType, *args: str | CppType | CppExpr) -> CppExpr:
    return CallExpr(name, args)


def multiline_call(name: str | CppType, *args: str | CppType | CppExpr) -> CppExpr:
    return CallExpr(name, args, multiline=True)


def render_cpp(value: str | CppType | CppExpr) -> str:
    if isinstance(value, CppExpr):
        return value.to_cpp()
    if isinstance(value, CppType):
        return value.to_cpp()
    return value


def render_namespace_type_aliases(types: dict[str, CppType]) -> str:
    return "\n".join(f"using {name} = {strip_cute_namespace(value.to_cpp())};" for name, value in types.items())


def strip_cute_namespace(cpp: str) -> str:
    return cpp.replace("cute::", "")


def cutlass_dtype_name(dtype: object) -> str:
    name = getattr(dtype, "__name__", None)
    if name is not None:
        return name
    return str(dtype)


def render_cutlass_dtype(dtype: object) -> str:
    name = cutlass_dtype_name(dtype)
    try:
        return CPP_DTYPE_NAMES[name]
    except KeyError as exc:
        raise ValueError(f"no C++ renderer registered for CUTLASS dtype {name}") from exc


def render_ir_dtype(dtype: str) -> str:
    try:
        return IR_DTYPE_NAMES[dtype]
    except KeyError as exc:
        raise ValueError(f"no C++ renderer registered for CuTe IR dtype {dtype}") from exc


def render_cute_tuple(value: Any, wrapper: str) -> str:
    if isinstance(value, int):
        return cute_int(value).to_cpp()
    if isinstance(value, CppType):
        return value.to_cpp()
    if isinstance(value, str):
        return value
    if not isinstance(value, tuple):
        raise TypeError(f"unsupported CuTe tuple item {value!r}")
    return f"cute::{wrapper}<" + ", ".join(render_cute_tuple(item, wrapper) for item in value) + ">"


def render_cute_tuple_expr(value: Any, wrapper: str) -> str:
    if isinstance(value, CppExpr):
        return value.to_cpp()
    if isinstance(value, int):
        return cute_int(value).instance()
    if isinstance(value, CppType):
        return value.instance()
    if isinstance(value, str):
        return f"{value}{{}}"
    if not isinstance(value, tuple):
        raise TypeError(f"unsupported CuTe tuple item {value!r}")
    factory = "make_shape" if wrapper == "Shape" else "make_stride"
    return f"cute::{factory}(" + ", ".join(render_cute_tuple_expr(item, wrapper) for item in value) + ")"


def render_layout_expr(shape: Any, stride: Any) -> str:
    return f"cute::make_layout({render_cute_tuple_expr(shape, 'Shape')}, {render_cute_tuple_expr(stride, 'Stride')})"


def render_layout(shape: Any, stride: Any) -> str:
    return f"decltype({render_layout_expr(shape, stride)})"


def render_composed_layout(
    swizzle: tuple[int, int, int],
    offset: int | str | CppType,
    outer_shape: Any,
    outer_stride: Any,
) -> str:
    swizzle_cpp = f"cute::Swizzle<{swizzle[0]}, {swizzle[1]}, {swizzle[2]}>{{}}"
    if isinstance(offset, int):
        offset_cpp = cute_int(offset).instance()
    elif isinstance(offset, CppType):
        offset_cpp = offset.instance()
    else:
        offset_cpp = f"{offset}{{}}"
    return f"decltype(cute::make_composed_layout({swizzle_cpp}, {offset_cpp}, {render_layout_expr(outer_shape, outer_stride)}))"



def tupleify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tupleify(item) for item in value)
    if isinstance(value, tuple):
        return tuple(tupleify(item) for item in value)
    return value


@contextmanager
def cutedsl_context():
    if _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize generated type aliases")
    with _ir.Context(), _ir.Location.unknown():
        yield


@lru_cache(maxsize=None)
def make_cutedsl_layout(shape: IntTuple, stride: IntTuple | None = None) -> Any:
    if _cute is None or _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize generated layout aliases")
    layout = _cute.make_layout(shape, stride=stride) if stride is not None else _cute.make_layout(shape)
    _CUTEDSL_KEEPALIVE.append(layout)
    return layout


def make_cutedsl_dynamic_layout(shape: LayoutTuple, stride: LayoutTuple) -> Any:
    if _cute is None or _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize generated dynamic layout aliases")
    layout = _cute.make_layout(
        _to_cutedsl_dynamic_tuple(shape),
        stride=_to_cutedsl_dynamic_tuple(stride),
    )
    _CUTEDSL_KEEPALIVE.append(layout)
    return layout


def _to_cutedsl_dynamic_tuple(value: LayoutTuple) -> Any:
    if isinstance(value, tuple):
        return tuple(_to_cutedsl_dynamic_tuple(item) for item in value)
    if isinstance(value, DynamicInt):
        from cutlass.base_dsl.typing import Int32

        return Int32(0)
    if isinstance(value, int):
        return value
    raise TypeError(f"unsupported dynamic CuTe layout item {value!r}")


def make_cutedsl_layout_shape_stride(shape: IntTuple, stride: IntTuple | None = None) -> tuple[Any, Any]:
    layout = make_cutedsl_layout(shape, stride)
    return tupleify(layout.shape), tupleify(layout.stride)


@lru_cache(maxsize=None)
def make_cutedsl_composed_layout(
    swizzle: tuple[int, int, int],
    offset: int,
    outer_shape: IntTuple,
    outer_stride: IntTuple,
) -> Any:
    if _cute is None or _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize generated composed layout aliases")
    swizzle_value = _cute.make_swizzle(*swizzle)
    outer = make_cutedsl_layout(outer_shape, outer_stride)
    layout = _cute.make_composed_layout(swizzle_value, offset, outer)
    _CUTEDSL_KEEPALIVE.extend([swizzle_value, layout])
    return layout


def cute_tile_to_shape(atom: CppType, target_shape: IntTuple, order: IntTuple) -> CppType:
    if _cute is None or _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize tile_to_shape layout aliases")
    dsl_value = getattr(atom, "dsl_value", None)
    if dsl_value is None:
        raise TypeError(f"{type(atom).__name__} cannot be used as a CuTe DSL tile_to_shape atom")
    layout = _cute.tile_to_shape(dsl_value, target_shape, order)
    _CUTEDSL_KEEPALIVE.append(layout)
    if isinstance(atom, CuteSmemLayoutAtom):
        return layout_type_from_cutedsl(
            layout,
            composed_hint=(atom.swizzle, f"cute::smem_ptr_flag_bits<{atom.flag_bits}>"),
        )
    return layout_type_from_cutedsl(layout)


def cute_blocked_product(atom: CppType, tiler_shape: LayoutTuple, tiler_stride: LayoutTuple) -> CppType:
    if _cute is None or _ir is None:
        raise RuntimeError("CuTe DSL is required to synthesize blocked_product layout aliases")
    dsl_value = getattr(atom, "dsl_value", None)
    if dsl_value is None:
        raise TypeError(f"{type(atom).__name__} cannot be used as a CuTe DSL blocked_product atom")
    tiler = make_cutedsl_dynamic_layout(tiler_shape, tiler_stride)
    layout = _cute.blocked_product(dsl_value, tiler)
    _CUTEDSL_KEEPALIVE.extend([tiler, layout])
    return layout_type_from_cutedsl(layout)


def layout_type_from_cutedsl(
    layout: Any,
    composed_hint: tuple[tuple[int, int, int], int | str | CppType] | None = None,
) -> CppType:
    type_text = str(layout.type)
    composed_type = re.fullmatch(r'!cute\.composed_layout<"(.*)">', type_text)
    layout_type = re.fullmatch(r'!cute\.layout<"(.*)">', type_text)
    if composed_type is not None:
        layout_text = composed_type.group(1)
    elif layout_type is not None:
        layout_text = layout_type.group(1)
    else:
        layout_text = str(layout)

    composed = re.fullmatch(r"S<(\d+),(\d+),(\d+)> o (-?\d+) o (.*):(.*)", layout_text)
    if composed is not None:
        outer_shape = _parse_cutedsl_layout_tuple(composed.group(5))
        outer_stride = _parse_cutedsl_layout_tuple(composed.group(6))
        if composed_hint is None:
            swizzle = tuple(int(composed.group(i)) for i in range(1, 4))
            offset: int | str | CppType = int(composed.group(4))
        else:
            swizzle, offset = composed_hint
        return CuteComposedLayout(
            layout,
            swizzle,
            offset,
            tupleify(outer_shape),
            tupleify(outer_stride),
        )

    shape, stride = _parse_layout_text(layout_text)
    return CuteConcreteLayout(layout, shape, stride)


@lru_cache(maxsize=None)
def make_cutedsl_tiled_mma_ir(atom_layout_shape: IntTuple, permutation_mnk: IntTuple) -> str:
    if _cute is None or _ir is None or _cutlass is None:
        raise RuntimeError("CuTe DSL is required to synthesize tiled MMA aliases")
    from cutlass.cute.nvgpu import warp

    atom_layout = _cute.make_layout(atom_layout_shape)
    op = warp.MmaMXF4NVF4Op(
        _cutlass.dtype("Float4E2M1FN"),
        _cutlass.dtype("Float32"),
        _cutlass.dtype("Float8E4M3FN"),
    )
    tiled_mma = _cute.make_tiled_mma(op, atom_layout, permutation_mnk)
    _CUTEDSL_KEEPALIVE.extend([atom_layout, op, tiled_mma])
    return str(tiled_mma._trait.value.type)


def make_cutedsl_copy_atom_ir(kind: str, element: CutlassDType) -> str:
    return _make_cutedsl_copy_atom_ir(kind, cutlass_dtype_name(element.value))


@lru_cache(maxsize=None)
def _make_cutedsl_copy_atom_ir(kind: str, element_name: str) -> str:
    if _cute is None or _ir is None or _cutlass is None:
        raise RuntimeError("CuTe DSL is required to synthesize copy atom aliases")
    from cutlass.cute.nvgpu import CopyUniversalOp, warp

    element_value = _cutlass.dtype(element_name)
    if kind == "ldsm_u32x4_n":
        op = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
    elif kind == "universal":
        op = CopyUniversalOp()
    else:
        raise ValueError(f"unsupported copy atom kind {kind}")
    atom = _cute.make_copy_atom(op, element_value)
    _CUTEDSL_KEEPALIVE.extend([op, atom])
    return str(atom._trait.value.type)


def tiled_mma(atom_layout_shape: IntTuple, permutation_mnk: IntTuple) -> CuteTiledMma:
    return CuteTiledMma(make_cutedsl_tiled_mma_ir(atom_layout_shape, permutation_mnk))


def copy_atom_ldsm_u32x4_n(element: CutlassDType) -> CuteCopyAtom:
    return CuteCopyAtom(make_cutedsl_copy_atom_ir("ldsm_u32x4_n", element))


def copy_atom_universal(element: CutlassDType) -> CuteCopyAtom:
    return CuteCopyAtom(make_cutedsl_copy_atom_ir("universal", element))


def _parse_layout_text(text: str) -> tuple[Any, Any]:
    shape_text, stride_text = text.split(":", 1)
    shape = _parse_cutedsl_layout_tuple(shape_text)
    stride = _parse_cutedsl_layout_tuple(stride_text)
    return tupleify(shape), tupleify(stride)


def _parse_cutedsl_layout_tuple(text: str) -> Any:
    if "?" not in text:
        return ast.literal_eval(text)

    parser = _LayoutTextParser(text)
    value = parser.parse_value()
    parser.expect_end()
    return value


class _LayoutTextParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse_value(self) -> Any:
        self._skip_space()
        if self._consume("("):
            items = []
            self._skip_space()
            if self._consume(")"):
                return tuple()
            while True:
                items.append(self.parse_value())
                self._skip_space()
                if self._consume(","):
                    self._skip_space()
                    if self._consume(")"):
                        break
                    continue
                self._expect(")")
                break
            return tuple(items)
        if self._consume("?"):
            if self._consume("{"):
                while self.pos < len(self.text) and self.text[self.pos] != "}":
                    self.pos += 1
                self._expect("}")
            return DynamicInt()
        return self._parse_int()

    def expect_end(self) -> None:
        self._skip_space()
        if self.pos != len(self.text):
            raise ValueError(f"trailing layout text at {self.pos}: {self.text!r}")

    def _parse_int(self) -> int:
        self._skip_space()
        start = self.pos
        if self.pos < len(self.text) and self.text[self.pos] == "-":
            self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == start or (self.pos == start + 1 and self.text[start] == "-"):
            raise ValueError(f"expected int at {self.pos}: {self.text!r}")
        return int(self.text[start:self.pos])

    def _consume(self, token: str) -> bool:
        self._skip_space()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._consume(token):
            raise ValueError(f"expected {token!r} at {self.pos}: {self.text!r}")

    def _skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1


def render_tiled_mma_ir(ir_type: str) -> str:
    mma = re.search(
        r"!cute_nvgpu\.SM120\.mma_bs<(?P<shape>\d+x\d+x\d+), vec_size = (?P<vec>\d+), "
        r"elem_type = \((?P<elem>[^)]*)\), sf_type = (?P<sf>[^,>]+)",
        ir_type,
    )
    atom_layout = re.search(r'atom_layout_MNK = <"(?P<layout>[^"]+)">', ir_type)
    permutation = re.search(r'permutation_MNK = <"\[(?P<tile>[^\]]+)\]">', ir_type)
    if mma is None or atom_layout is None or permutation is None:
        raise ValueError(f"unsupported CuTe tiled MMA IR type: {ir_type}")
    if mma.group("shape") != "16x8x64" or mma.group("vec") != "16":
        raise ValueError(f"unsupported SM120 block-scaled MMA shape in IR: {ir_type}")
    elem_types = [item.strip() for item in mma.group("elem").split(",")]
    if elem_types != ["f4E2M1FN", "f4E2M1FN", "f32"] or mma.group("sf").strip() != "f8E4M3FN":
        raise ValueError(f"unsupported SM120 block-scaled MMA element types in IR: {ir_type}")

    layout_shape, layout_stride = _parse_layout_text(atom_layout.group("layout"))
    tile_modes = []
    for mode in permutation.group("tile").split(";"):
        extent, stride = mode.split(":", 1)
        if stride != "1":
            raise ValueError(f"unsupported non-unit tiled MMA permutation stride in IR: {ir_type}")
        tile_modes.append(cute_int(int(extent)))
    tile_cpp = template_type("cute::Tile", *tile_modes).to_cpp()
    return decltype(
        multiline_call(
            "cute::make_tiled_mma",
            instance("cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4"),
            instance(named(render_layout(layout_shape, layout_stride))),
            instance(named(tile_cpp)),
        )
    ).to_cpp()


def render_copy_atom_ir(ir_type: str) -> str:
    universal = re.search(r"!cute_nvgpu\.atom\.universal_copy<(?P<dtype>[^>]+)>", ir_type)
    if universal is not None:
        dtype = render_ir_dtype(universal.group("dtype").strip())
        return template_type("cute::Copy_Atom", template_type("cute::UniversalCopy", dtype), dtype).to_cpp()

    ldsm = re.search(
        r"!cute_nvgpu\.atom\.ldsm<val_type = (?P<dtype>[^,]+), mode = <\"\(8,8\)\">, "
        r"sz_pattern = u16, num_matrices = 4, n>",
        ir_type,
    )
    if ldsm is not None:
        return template_type("cute::Copy_Atom", "cute::SM75_U32x4_LDSM_N", render_ir_dtype(ldsm.group("dtype").strip())).to_cpp()

    raise ValueError(f"unsupported CuTe copy atom IR type: {ir_type}")


def blockscaled_sf_atom(sf_vec_size: int) -> CuteLayout:
    return cute_layout(((16, 4), (sf_vec_size, 4)), stride=((16, 4), (0, 1)))


def blockscaled_smem_layout_atom(row_extent: int, vec_extent: int, sf_vec_size: int = 16) -> CuteLayout:
    mma_nsf = 4
    blk_mn = 64
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    row_blocks = row_extent // blk_mn
    vec_blocks = vec_extent // sf_vec_size // blk_sf
    shape_mn = ((16, 4), row_blocks)
    shape_k = ((sf_vec_size, mma_nsf), blk_sf // mma_nsf, vec_blocks)
    stride_mn = ((16, 4), blk_elems)
    stride_k = ((0, 1), mma_nsf, row_blocks * blk_elems)
    return cute_layout((shape_mn, shape_k), stride=(stride_mn, stride_k))


def lambda_smem_layout_atom(row_extent: int) -> CuteLayout:
    row_block = 4
    return cute_layout((row_extent // row_block, row_block), stride=(row_block, 1))


def flatten_layout(shape: Any, stride: Any) -> list[tuple[int, int]]:
    if isinstance(shape, int):
        return [(shape, int(stride))]
    result: list[tuple[int, int]] = []
    for shape_item, stride_item in zip(shape, stride, strict=True):
        result.extend(flatten_layout(shape_item, stride_item))
    return result


def layout_cosize(shape: Any, stride: Any) -> int:
    flat = flatten_layout(shape, stride)
    return 1 + sum((extent - 1) * stride for extent, stride in flat if stride > 0)


def append_stage_layout(layout: CuteLayout, stages: int) -> CuteLayout:
    stage_stride = layout_cosize(layout.dsl_shape, layout.dsl_stride)
    return cute_layout(tuple(layout.dsl_shape) + (stages,), stride=tuple(layout.dsl_stride) + (stage_stride,))
