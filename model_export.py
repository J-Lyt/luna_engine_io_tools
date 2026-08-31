from operator import itemgetter

from .utils import *
from .constants import SUBSET_CENTER_LOG_SCALE
from .dat1 import DAT1_BLOCK_TABLE_ENTRY_SIZE, DAT1_FILE_ID, DAT1_FIXUP_TABLE_ENTRY_SIZE, DAT1_HEADER_SIZE
from .hashes import BLOCK_HASHES, string_crc32
from .model_morph import (
    MORPH_DELTA_PRECISION,
    MORPH_VERTEX_DELTA_EPSILON,
    decode_model_morph2,
    encode_model_morph2,
    encode_model_smooth2,
)
from .model_msmr import (
    MSMR_MODEL_ANIM_MORPH_DATA_HASH,
    MSMR_MODEL_ANIM_MORPH_INDICES_HASH,
    MSMR_MODEL_ANIM_MORPH_INFO_HASH,
    MSMR_MODEL_COL_VERT_HASH,
    MSMR_MODEL_INDEX_HASH,
    MSMR_MODEL_MAGIC,
    MSMR_MODEL_SKIN_BATCH_HASH,
    MSMR_MODEL_SKIN_DATA_HASH,
    MSMR_MODEL_SKIN_JOINT_REMAP_HASH,
    MSMR_MODEL_STD_VERT_HASH,
    MSMR_MODEL_UV1_VERT_HASH,
    MSMR_SUBSET_RECORD_SIZE,
    decode_msmr_geometry,
    decode_msmr_morphs,
    decode_msmr_skin_weights,
    encode_msmr_index_stream,
    encode_msmr_morphs,
    encode_msmr_vertex_stream,
    is_msmr_model,
    parse_msmr_look_groups_metadata,
    parse_msmr_subset,
)
from .model_import import (
    MODEL_MATERIAL_INFO_SIZE,
    MODEL_MATERIAL_SIZE,
    MODEL_SUBSET_BASE_OFFSET,
    MODEL_SUBSET_FLAGS_OFFSET,
    MODEL_SUBSET_INDEX_COUNT_OFFSET,
    MODEL_SUBSET_INDEX_DATA_OFFSET,
    MODEL_SUBSET_MATERIAL_INDEX_OFFSET,
    MODEL_SUBSET_MPU_OFFSET,
    MODEL_SUBSET_RECORD_SIZE,
    MODEL_SUBSET_UV_LOG_OFFSET,
    MODEL_SUBSET_VERTEX_COUNT_OFFSET,
    MODEL_SUBSET_VERTEX_STD_OFFSET,
    MODEL_SUBSET_VERTEX_UV12_OFFSET,
    MODEL_UV_FLOAT_TO_FIXED_BASE,
    SKIN_CLUSTER_FULL_INDEX_BIT,
    SKIN_CLUSTER_INFLUENCE_SHIFT,
    SKIN_CLUSTER_JOINT_OFFSET_SHIFT,
    SKIN_CLUSTER_OFFSET_MASK,
    SKIN_CLUSTER_VERTEX_COUNT,
    SKIN_CLUSTER_WORD_BYTES,
    SKIN_WEIGHT_SCALE,
    SUBSET_FLAG_HAS_UV1,
    SUBSET_FLAG_HAS_UV2,
    SUBSET_FLAG_SKINNED,
    _decode_packed_normal,
    _decode_packed_tangent,
    _parse_model_materials,
    _read_c_string,
)

DAT1_BLOCK_ALIGN = 16
DAT1_CACHELINE_ALIGN = 64
MSMR_SKIN_BATCH_MAX_VERTEX_COUNT = 2560
MSMR_LOOK_LOD_COUNT = 6
MSMR_LOOK_BUILT_LOD_MASK_SIZE = 256
STG_MAGIC = 0x00475453
STG_VERSION = 0x1
STG_HEADER_ALIGN = 16

MODEL_BUILT_SIZE = 96
MODEL_LOOK_SIZE = 64
MODEL_BVH_RECORD_SIZE = 184  # native per-look ModelLookBVHInfo record size (see _build_inert_look_bvh_blocks)
MODEL_LOOK_BUILT_SIZE = 80
MODEL_LOOK_GROUP_SIZE = 24
MODEL_STD_VERTEX_SIZE = 16
MODEL_MAX_VERTEX_COUNT = 0xFFFF
MODEL_SPLIT_VERTEX_TARGET = 60000
MODEL_DEFAULT_MPU = 1.0 / 32768.0

SUBSET_FLAG_HAS_ANIM_VERT = 0x0002
SUBSET_FLAG_HAS_COLOR = 0x0010
SUBSET_FLAG_HAS_ORIGIN_OFFSET = 0x4000
SUBSET_EXPORT_FLAG_CLEAR_MASK = (
    SUBSET_FLAG_SKINNED
    | SUBSET_FLAG_HAS_ANIM_VERT
    | SUBSET_FLAG_HAS_UV1
    | SUBSET_FLAG_HAS_UV2
    | SUBSET_FLAG_HAS_COLOR
    | SUBSET_FLAG_HAS_ORIGIN_OFFSET
)

MODEL_FLAG_HAS_SKINNING = 1 << 1
MODEL_FLAG_HAS_GPU_SKINNING = 1 << 2
MODEL_FLAG_ANIM_VERT = 1 << 28
MODEL_FLAG_ANIM_DYNAMICS = 1 << 29
MODEL_FLAG_USES_AUTO_LODS = 1 << 30

SKIN_JOINT_OFFSET_STEP = 256
SKIN_JOINT_OFFSET_MAX = SKIN_JOINT_OFFSET_STEP * 15
SKIN_UINT8_MAX = 255
SKIN_CLUSTER_ANIM_VERT_BIT = 1 << 29

MODEL_BUILT_FLAGS_OFFSET = 0
MODEL_BUILT_FADE_OUT_DIST_OFFSET = 24
MODEL_BUILT_BSPHERE_OFFSET = 32
MODEL_BUILT_AABB_EXTENTS_OFFSET = 48
MODEL_BUILT_COMMON_MPU_OFFSET = 60
MODEL_BUILT_VERTEX_MPU_OFFSET = 64
MODEL_BUILT_CUSTOM_STREAM_COUNT_OFFSET = 68
MODEL_BUILT_CONTENT_FLAGS_OFFSET = 72
MODEL_BUILT_SUBSET_LOD_MASK_COUNT_OFFSET = 74
MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET = 78

CONTENT_FLAG_ANIM_MORPH = 0x0001
CONTENT_FLAG_ANIM_ZIVA = 0x0004
CONTENT_FLAG_ANIM_VERT_SMOOTH = 0x0010
CONTENT_FLAG_USES_AUTO_LODS = 0x0080

MODEL_SUBSET_SURFACE_AREA_OFFSET = 32
MODEL_SUBSET_UV_AREA_OFFSET = 36
MODEL_SUBSET_FADE_OUT_DIST_OFFSET = 40
MODEL_SUBSET_MATERIAL_LOD_DIST_OFFSET = 44
MODEL_SUBSET_OBJ_CENTER_OFFSET = 48
MODEL_SUBSET_OBJ_EXTENTS_OFFSET = 60
MODEL_SUBSET_LONGEST_EDGE_OFFSET = 118
MODEL_SUBSET_CURVATURE_RADIUS_OFFSET = 120
MODEL_GLOBAL_BOUNDS_PADDING = 0.05

ASSET_CHUNK_UNCOMPRESSED_MASK = (1 << 30) - 1
ASSET_CHUNK_COMPRESSED_SHIFT = 30
ASSET_CHUNK_COMPRESSION_SHIFT = 60
ASSET_COMPRESSION_NONE = 0


def _align(value, alignment):
    return (int(value) + alignment - 1) & ~(alignment - 1)


def _align_buffer(buffer, alignment):
    pad = _align(len(buffer), alignment) - len(buffer)
    if pad:
        buffer += b"\x00" * pad


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _clamp_i16(value):
    return int(_clamp(int(round(value)), -32768, 32767))


def _clamp_u16(value):
    return int(_clamp(int(round(value)), 0, 65535))


def _clamp_u8(value):
    return int(_clamp(int(round(value)), 0, 255))


def _round_engine(value):
    value = float(value)
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_len(a):
    return math.sqrt(max(0.0, _vec_dot(a, a)))


def _vec_normalize(a, fallback=(0.0, 0.0, 1.0)):
    length = _vec_len(a)
    if length <= 1e-8:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def _luna_triangle_tangent_space(positions, uvs):
    #matches luna tangent gen
    if len(positions) != 3 or len(uvs) != 3:
        raise ValueError("Luna tangent generation requires one triangle")
    epsilon = 0.000001
    edge_01 = _vec_sub(positions[1], positions[0])
    edge_02 = _vec_sub(positions[2], positions[0])
    triangle_normal = _vec_normalize(_vec_cross(edge_01, edge_02), fallback=(0.0, 1.0, 0.0))

    min_v, mid_v, max_v = 0, 1, 2
    if uvs[max_v][1] < uvs[mid_v][1]:
        max_v, mid_v = mid_v, max_v
    if uvs[max_v][1] < uvs[min_v][1]:
        max_v, min_v = min_v, max_v
    if uvs[mid_v][1] < uvs[min_v][1]:
        mid_v, min_v = min_v, mid_v
    v_range = float(uvs[max_v][1]) - float(uvs[min_v][1])
    interp = (
        (float(uvs[mid_v][1]) - float(uvs[min_v][1])) / v_range
        if v_range > epsilon else 1.0
    )
    interp_pos = _vec_add(
        _vec_mul(positions[min_v], 1.0 - interp),
        _vec_mul(positions[max_v], interp),
    )
    interp_u = float(uvs[min_v][0]) * (1.0 - interp) + float(uvs[max_v][0]) * interp
    tangent = _vec_sub(interp_pos, positions[mid_v])
    if interp_u < float(uvs[mid_v][0]):
        tangent = _vec_mul(tangent, -1.0)
    tangent = _vec_normalize(tangent, fallback=(1.0, 0.0, 0.0))

    min_u, mid_u, max_u = 0, 1, 2
    if uvs[max_u][0] < uvs[mid_u][0]:
        max_u, mid_u = mid_u, max_u
    if uvs[max_u][0] < uvs[min_u][0]:
        max_u, min_u = min_u, max_u
    if uvs[mid_u][0] < uvs[min_u][0]:
        mid_u, min_u = min_u, mid_u
    u_range = float(uvs[max_u][0]) - float(uvs[min_u][0])
    interp = (
        (float(uvs[mid_u][0]) - float(uvs[min_u][0])) / u_range
        if u_range > epsilon else 1.0
    )
    interp_pos = _vec_add(
        _vec_mul(positions[min_u], 1.0 - interp),
        _vec_mul(positions[max_u], interp),
    )
    interp_v = float(uvs[min_u][1]) * (1.0 - interp) + float(uvs[max_u][1]) * interp
    binormal = _vec_sub(interp_pos, positions[mid_u])
    if interp_v < float(uvs[mid_u][1]):
        binormal = _vec_mul(binormal, -1.0)
    binormal = _vec_normalize(binormal, fallback=(0.0, 0.0, 1.0))

    computed_binormal = _vec_cross(tangent, triangle_normal)
    tangent_flip = 1.0 if _vec_dot(computed_binormal, binormal) > 0.0 else -1.0
    return tangent, tangent_flip


def _blender_to_engine_vec(value):
    return (float(value.x), float(value.z), -float(value.y))


def _safe_matrix_relative_to_armature(arm, obj):
    try:
        if arm:
            return arm.matrix_world.inverted() @ obj.matrix_world
        return obj.matrix_world.copy()
    except Exception:
        return getattr(obj, "matrix_local", mathutils.Matrix.Identity(4))


def _source_path_from_armature(arm):
    try:
        path = str(arm.get("engine_model_source_path", "") or "")
    except Exception:
        path = ""
    return path


def _resolve_model_armature(context):
    arm = _resolve_anim_armature(context)
    if arm:
        return arm
    active = getattr(context, "active_object", None)
    if active and getattr(active, "type", None) == "ARMATURE":
        return active
    if active and getattr(active, "parent", None) and active.parent.type == "ARMATURE":
        return active.parent
    return None


def _parse_u64(value, default=0):
    if value is None:
        return int(default) & U64_MASK
    if isinstance(value, int):
        return int(value) & U64_MASK
    text = str(value).strip()
    if not text:
        return int(default) & U64_MASK
    try:
        return int(text, 0) & U64_MASK
    except ValueError:
        return int(default) & U64_MASK


def _valid_material_id(index):
    return (0x8000000000000000 | ((int(index) + 1) & 0x3FFFFFFFFFFFFFFF)) & U64_MASK


def _normalize_material_asset_path(path):
    text = str(path or "").strip()
    text = re.sub(r"(?i)(\.material)\.\d{3}$", r"\1", text)
    if text and not text.lower().endswith(".material"):
        text = f"{text}.material"
    return text


class _StringPool:
    def __init__(self, base_offset, initial_bytes):
        self.base_offset = int(base_offset)
        self.buffer = bytearray(initial_bytes or b"")
        self.offsets = {}
        self._index_existing_strings()

    def _index_existing_strings(self):
        start = 0
        data = bytes(self.buffer)
        while start < len(data):
            end = data.find(b"\x00", start)
            if end < 0:
                break
            if end > start:
                try:
                    text = data[start:end].decode("ascii")
                except UnicodeDecodeError:
                    text = ""
                if text and text not in self.offsets:
                    self.offsets[text] = self.base_offset + start
            start = end + 1

    def add(self, text):
        text = str(text or "")
        if text in self.offsets:
            return self.offsets[text]
        encoded = text.encode("ascii", errors="ignore") + b"\x00"
        offset = self.base_offset + len(self.buffer)
        self.buffer += encoded
        self.offsets[text] = offset
        return offset

    def bytes(self):
        return bytes(self.buffer)


class _Dat1Template:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "rb") as f:
            raw = f.read()
        dat1_offset = raw.find(b"1TAD")
        if dat1_offset < 0:
            raise ValueError("DAT1 magic not found")
        self.had_stg = raw[:4] == struct.pack("<I", STG_MAGIC)
        self.prefix = raw[:dat1_offset]
        self.data = raw[dat1_offset:]
        if len(self.data) < DAT1_HEADER_SIZE:
            raise ValueError("DAT1 header is truncated")
        data_file_id, version, declared_size, block_count, fixup_count = struct.unpack_from("<IIIHH", self.data, 0)
        if data_file_id != DAT1_FILE_ID:
            raise ValueError("invalid DAT1 file id")
        if declared_size > len(self.data):
            raise ValueError("DAT1 declared size is larger than the source file")
        self.data = self.data[:declared_size]
        self.version = version
        self.block_count = block_count
        self.fixup_count = fixup_count
        self.sb_offset = (
            DAT1_HEADER_SIZE
            + block_count * DAT1_BLOCK_TABLE_ENTRY_SIZE
            + fixup_count * DAT1_FIXUP_TABLE_ENTRY_SIZE
        )
        self.entries = []
        self.blocks = {}
        cursor = DAT1_HEADER_SIZE
        for _index in range(block_count):
            name_hash, block_offset, block_size = struct.unpack_from("<III", self.data, cursor)
            self.entries.append((name_hash, block_offset, block_size))
            self.blocks[name_hash] = (block_offset, block_size)
            cursor += DAT1_BLOCK_TABLE_ENTRY_SIZE
        self.fixup_table = self.data[cursor:self.sb_offset]
        first_block = min((off for _hash, off, size in self.entries if size or off), default=len(self.data))
        self.string_buffer = self.data[self.sb_offset:first_block]

    def payload(self, block_hash):
        off, size = self.blocks[block_hash]
        return self.data[off:off + size]


def _material_from_blender(mat, fallback, index, preserve_fallback_identity=False):
    fallback = fallback or {}
    path = str(getattr(mat, "engine_material_path", "") or "")
    if not path:
        mat_name = str(getattr(mat, "name", "") or "")
        if ".material" in mat_name.lower() or "\\" in mat_name or "/" in mat_name:
            path = mat_name
    path = _normalize_material_asset_path(path or fallback.get("path", "") or "")
    fallback_path = _normalize_material_asset_path(fallback.get("path", "") or "")
    if preserve_fallback_identity and fallback_path and _material_paths_match(fallback_path, path):
        mapping = str(fallback.get("mapping", "") or "")
        material_id = int(fallback.get("id", 0) or 0) & U64_MASK
        flags = int(fallback.get("flags", 0) or 0) & U32_MASK
        mapping_hash = int(fallback.get("mapping_hash", 0)) & U32_MASK
        path = fallback_path
    else:
        mapping = str(getattr(mat, "engine_material_mapping_name", "") or fallback.get("mapping", "") or "")
        material_id = _parse_u64(getattr(mat, "get", lambda *_args: None)("engine_material_id"), fallback.get("id", 0))
        flags = int(getattr(mat, "get", lambda *_args: 0)("engine_material_flags", fallback.get("flags", 0)) or 0) & U32_MASK
        mapping_hash = int(fallback.get("mapping_hash", 0)) & U32_MASK
        if mapping:
            mapping_hash = string_crc32(mapping)
    return {
        "index": index,
        "path": path,
        "mapping": mapping,
        "id": material_id or _valid_material_id(index),
        "mapping_hash": mapping_hash,
        "flags": flags,
    }


def _material_asset_path_from_blender(mat):
    path = _normalize_material_asset_path(getattr(mat, "engine_material_path", "") or "")
    if not path:
        mat_name = str(getattr(mat, "name", "") or "")
        if ".material" in mat_name.lower() or "\\" in mat_name or "/" in mat_name:
            path = _normalize_material_asset_path(mat_name)
    return path


def _material_paths_match(left, right):
    left = _normalize_material_asset_path(left or "")
    right = _normalize_material_asset_path(right or "")
    return bool(left and right and left.lower() == right.lower())


def _primary_material_for_object(obj):
    mesh = getattr(obj, "data", None)
    materials = list(getattr(mesh, "materials", []) or []) if mesh else []
    if not materials:
        return None
    polygons = getattr(mesh, "polygons", None)
    polygon_count = len(polygons) if polygons is not None else 0
    if polygon_count:
        material_indices = np.empty(polygon_count, dtype=np.int32)
        polygons.foreach_get("material_index", material_indices)
        values, first_seen, counts = np.unique(
            material_indices, return_index=True, return_counts=True
        )
        # ?
        appearance = np.argsort(first_seen)
        index = int(values[appearance[int(np.argmax(counts[appearance]))]])
        if 0 <= index < len(materials):
            return materials[index]
    return materials[0]


def _set_material_index_prop(mat, material_index):
    try:
        mat["engine_material_index"] = int(material_index)
    except Exception:
        pass


def _set_object_material_index_prop(obj, material_index):
    try:
        obj["engine_material_index"] = int(material_index)
    except Exception:
        pass


def _build_material_entries(mesh_objects, original_materials, export_warnings=None):
    entries = [dict(entry) for entry in original_materials]
    if not entries:
        entries.append({
            "index": 0,
            "path": "",
            "mapping": "default",
            "id": _valid_material_id(0),
            "mapping_hash": string_crc32("default"),
            "flags": 0,
        })
    assigned_indices = []
    claimed_slots = {}
    for obj in mesh_objects:
        mat = _primary_material_for_object(obj)
        fallback_index = int(obj.get("engine_material_index", 0) or 0)
        if not mat:
            assigned_index = _clamp(fallback_index, 0, len(entries) - 1)
            assigned_indices.append(assigned_index)
            _set_object_material_index_prop(obj, assigned_index)
            _append_export_warning(
                export_warnings,
                f"{obj.name} has no Blender material. The original game material was kept. "
                "Assign a material only if you want to replace it.",
            )
            continue

        material_index_prop = mat.get("engine_material_index")
        material_index = None
        try:
            material_index = int(material_index_prop)
        except Exception:
            material_index = None
        
        path = _material_asset_path_from_blender(mat)
        mapping = str(getattr(mat, "engine_material_mapping_name", "") or "")

        if material_index is not None and material_index >= 0:
            existing_path = entries[material_index].get("path", "") if material_index < len(entries) else ""
            slot_is_safe = not existing_path or not path or _material_paths_match(existing_path, path)
            if material_index in claimed_slots and claimed_slots[material_index] != mat.name:
                slot_is_safe = False
            if slot_is_safe:
                while material_index >= len(entries):
                    entries.append({
                        "index": len(entries),
                        "path": "",
                        "mapping": f"material_{len(entries):03d}",
                        "id": _valid_material_id(len(entries)),
                        "mapping_hash": 0,
                        "flags": 0,
                    })
                entries[material_index] = _material_from_blender(
                    mat,
                    entries[material_index],
                    material_index,
                    preserve_fallback_identity=bool(existing_path and path and _material_paths_match(existing_path, path)),
                )
                claimed_slots[material_index] = mat.name
                _set_material_index_prop(mat, material_index)
                _set_object_material_index_prop(obj, material_index)
                assigned_indices.append(material_index)
                continue
            material_index = None

        if not path:
            _append_export_warning(
                export_warnings,
                f"{obj.name}'s material '{mat.name}' is not linked to a game material. The original game "
                "settings were kept. Set Material Asset Path in the Luna Engine Material panel if you want "
                "to replace them.",
            )
        match_index = None
        for idx, entry in enumerate(entries):
            if path and _material_paths_match(entry.get("path", ""), path):
                match_index = idx
                break
            if not path and mapping and entry.get("mapping") == mapping:
                match_index = idx
                break
        if match_index is None:
            match_index = len(entries)
            entries.append(_material_from_blender(mat, None, match_index))
        else:
            entries[match_index] = _material_from_blender(
                mat,
                entries[match_index],
                match_index,
                preserve_fallback_identity=bool(path and _material_paths_match(entries[match_index].get("path", ""), path)),
            )
        _set_material_index_prop(mat, match_index)
        _set_object_material_index_prop(obj, match_index)
        assigned_indices.append(match_index)
    return entries, assigned_indices


def _build_material_block(material_entries, string_pool):
    info_bytes = bytearray()
    runtime_bytes = bytearray()
    for entry in material_entries:
        path_offset = string_pool.add(entry.get("path", ""))
        mapping_offset = string_pool.add(entry.get("mapping", ""))
        info_bytes += struct.pack("<IIII", path_offset, 0, mapping_offset, 0)
    for index, entry in enumerate(material_entries):
        mapping = str(entry.get("mapping", "") or "")
        mapping_hash = int(entry.get("mapping_hash", 0)) & U32_MASK
        if mapping_hash == 0 and mapping:
            mapping_hash = string_crc32(mapping)
        material_id = int(entry.get("id", 0)) & U64_MASK
        if material_id == 0:
            material_id = _valid_material_id(index)
        runtime_bytes += struct.pack("<QII", material_id, mapping_hash, int(entry.get("flags", 0)) & U32_MASK)
    return bytes(info_bytes + runtime_bytes)


def _append_export_warning(warnings, message):
    if warnings is None:
        return
    message = str(message or "").strip()
    if message and message not in warnings:
        warnings.append(message)


def _format_export_warnings(warnings, limit=5):
    shown = list(warnings[:limit])
    if len(warnings) > limit:
        shown.append(
            f"After fixing these, export again to see the remaining {len(warnings) - limit} check(s)"
        )
    return "Export finished. Please check: " + " | ".join(shown)


def _friendly_export_error(exc):
    message = str(exc or "").strip()
    if isinstance(exc, ValueError) and message:
        return message
    return "Something unexpected stopped the export. Re-import the original .model file and try again."


def _uv_name_slot(name):
    clean = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    if clean in {"uv0", "uvmap", "map1", "texcoord0", "texturecoordinate0"}:
        return 0
    if clean in {"uv1", "uvmap1", "map2", "texcoord1", "texturecoordinate1"}:
        return 1
    if clean in {"uv2", "uvmap2", "map3", "texcoord2", "texturecoordinate2"}:
        return 2
    return None


def _uv_layer_has_nonzero(layer):
    try:
        for item in layer.data:
            uv = item.uv
            if abs(float(uv.x)) > 1e-7 or abs(1.0 - float(uv.y)) > 1e-7:
                return True
    except Exception:
        pass
    return False


def _zero_uv_layer(layer):
    try:
        for item in layer.data:
            item.uv = (0.0, 1.0)
    except Exception:
        pass


def _uv_layers_match(left_layer, right_layer):
    if not left_layer or not right_layer:
        return False
    try:
        if len(left_layer.data) != len(right_layer.data):
            return False
        for left_item, right_item in zip(left_layer.data, right_layer.data):
            left_uv = left_item.uv
            right_uv = right_item.uv
            if abs(float(left_uv.x) - float(right_uv.x)) > 1e-7:
                return False
            if abs(float(left_uv.y) - float(right_uv.y)) > 1e-7:
                return False
    except Exception:
        return False
    return True


def _ensure_model_uv_layers(obj, warnings=None):
    mesh = getattr(obj, "data", None)
    uv_layers = getattr(mesh, "uv_layers", None) if mesh else None
    result = {
        0: {"layer": None, "source_present": False},
        1: {"layer": None, "source_present": False},
        2: {"layer": None, "source_present": False},
    }
    if uv_layers is None:
        return result

    original_layers = list(uv_layers)
    used = []

    def is_used(layer):
        return any(layer == existing for existing in used)

    def pick_named(slot):
        for layer in original_layers:
            if not is_used(layer) and _uv_name_slot(layer.name) == slot:
                return layer
        return None

    for slot in range(3):
        layer = pick_named(slot)
        if layer is not None:
            used.append(layer)
            result[slot] = {"layer": layer, "source_present": True}

    for slot in range(3):
        if result[slot]["layer"] is not None:
            continue
        if slot < len(original_layers) and not is_used(original_layers[slot]):
            layer = original_layers[slot]
            used.append(layer)
            result[slot] = {"layer": layer, "source_present": True}

    for slot, info in result.items():
        layer = info["layer"]
        if layer is not None and layer.name != f"UV{slot}":
            old_name = layer.name
            layer.name = f"__LunaUV{slot}"
            _append_export_warning(
                warnings,
                f"{obj.name}'s UV map '{old_name}' was renamed to 'UV{slot}' so the game can read it. "
                "No action is needed unless you meant to use a different UV map.",
            )

    for slot, info in result.items():
        layer = info["layer"]
        if layer is None:
            layer = uv_layers.new(name=f"UV{slot}")
            _zero_uv_layer(layer)
            result[slot] = {"layer": layer, "source_present": False}
        layer.name = f"UV{slot}"

    if not original_layers:
        _append_export_warning(
            warnings,
            f"{obj.name} had no UV maps, so blank ones were added. Export can finish, but textures may "
            "look wrong. Unwrap the mesh to UV0, then export again.",
        )
    return result


def _should_export_uv_channel(obj, uv_info, slot, original_flags):
    flag = SUBSET_FLAG_HAS_UV1 if slot == 1 else SUBSET_FLAG_HAS_UV2
    if original_flags & flag:
        return True
    layer = uv_info.get(slot, {}).get("layer")
    prop_name = f"engine_uv{slot}_present"
    if prop_name in obj:
        return bool(obj.get(prop_name, False))
    if not _uv_layer_has_nonzero(layer):
        return False
    uv0_layer = uv_info.get(0, {}).get("layer")
    if uv0_layer is not None and _uv_layers_match(layer, uv0_layer):
        return False
    return bool(uv_info.get(slot, {}).get("source_present", False)) or True


def _uv_layer_by_name_or_index(mesh, name, index):
    try:
        layer = mesh.uv_layers.get(name)
        if layer:
            return layer
    except Exception:
        pass
    try:
        if index < len(mesh.uv_layers):
            return mesh.uv_layers[index]
    except Exception:
        pass
    return None


def _linear_matrix_is_identity(matrix, threshold=1.0e-7):
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if abs(float(matrix[row][column]) - expected) > threshold:
                return False
    return True


def _source_model_basis(mesh, uv0_layer, basis_coords):
    vertex_count = len(mesh.vertices)
    attributes = {}
    for name in (
        "engine_source_normal_tangent",
        "engine_position_w",
        "engine_source_position",
        "engine_source_uv0_u",
        "engine_source_uv0_v",
    ):
        attribute = mesh.attributes.get(name)
        if attribute is None or attribute.domain != 'POINT' or len(attribute.data) != vertex_count:
            return None
        attributes[name] = attribute.data

    expected_signature = str(mesh.get("engine_source_topology_signature", "") or "")
    current_signature = model_topology_signature(mesh_triangle_vertex_indices(mesh))
    if not expected_signature or current_signature != expected_signature:
        return None

    expected_normal_signature = str(
        mesh.get("engine_source_corner_normal_signature", "") or ""
    )
    if expected_normal_signature and model_corner_normal_signature(mesh.corner_normals) != expected_normal_signature:
        return None

    source_positions = attributes["engine_source_position"]
    source_position_values = np.empty(vertex_count * 3, dtype=np.float64)
    source_positions.foreach_get("vector", source_position_values)
    if basis_coords is not None:
        current_positions = np.asarray(basis_coords, dtype=np.float64).reshape(-1)
    else:
        current_positions = np.empty(vertex_count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", current_positions)
    if np.any(np.abs(current_positions - source_position_values) > 1.0e-7):
        return None

    source_uv0_u = attributes["engine_source_uv0_u"]
    source_uv0_v = attributes["engine_source_uv0_v"]
    loop_count = len(mesh.loops)
    loop_vertex_indices = np.empty(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    source_u = np.empty(vertex_count, dtype=np.float64)
    source_v = np.empty(vertex_count, dtype=np.float64)
    source_uv0_u.foreach_get("value", source_u)
    source_uv0_v.foreach_get("value", source_v)

    if uv0_layer:
        current_uv_flat = np.empty(loop_count * 2, dtype=np.float64)
        uv0_layer.data.foreach_get("uv", current_uv_flat)
        current_u = current_uv_flat[0::2]
        current_v = 1.0 - current_uv_flat[1::2]
    else:
        current_u = np.zeros(loop_count, dtype=np.float64)
        current_v = np.zeros(loop_count, dtype=np.float64)

    if np.any(np.abs(current_u - source_u[loop_vertex_indices]) > 1.0e-7) or np.any(
        np.abs(current_v - source_v[loop_vertex_indices]) > 1.0e-7
    ):
        return None

    if not expected_normal_signature:
        source_words = np.empty(vertex_count, dtype=np.int32)
        attributes["engine_source_normal_tangent"].foreach_get("value", source_words)
        decoded = _decode_packed_normal_array(source_words.astype(np.uint32))
        source_normals = np.empty_like(decoded)
        source_normals[:, 0] = decoded[:, 0]
        source_normals[:, 1] = -decoded[:, 2]
        source_normals[:, 2] = decoded[:, 1]

        current_normals = np.empty(loop_count * 3, dtype=np.float64)
        mesh.corner_normals.foreach_get("vector", current_normals)
        current_normals = current_normals.reshape(-1, 3)

        # Blender normalizes custom split normals when assigning them
        dots = np.einsum("ij,ij->i", current_normals, source_normals[loop_vertex_indices])
        normal_mismatch_limit = max(4, int(loop_count * 0.001))
        if int(np.count_nonzero(dots < 0.99999)) > normal_mismatch_limit:
            return None

    return attributes


VERTEX_MATCH_NORMAL_DOT_MIN = 1.0 - 0.002
VERTEX_MATCH_TANGENT_DOT_MIN = 1.0 - 1.6
VERTEX_MATCH_UV_THRESHOLD = 0.0001


def _vertex_group_joint_map(obj, arm):
    if not arm:
        return {}
    bone_names = {}
    for bone in getattr(arm.data, "bones", []) or []:
        try:
            engine_index = int(bone.get("engine_joint_index", -1))
        except Exception:
            engine_index = -1
        bone_names[bone.name] = engine_index
    result = {}
    for group in getattr(obj, "vertex_groups", []) or []:
        if group.name in bone_names:
            result[group.index] = bone_names[group.name]
    return result


def _vertex_weights(mesh_vertex, obj, group_to_joint, source_joint_count):
    weights = []
    lookup_joint = group_to_joint.get
    for group_elem in getattr(mesh_vertex, "groups", []) or []:
        joint_index = lookup_joint(int(group_elem.group))
        if joint_index is None:
            continue
        weight = float(group_elem.weight)
        if weight > 0.0:
            if source_joint_count is not None and not 0 <= joint_index < source_joint_count:
                raise ValueError(
                    f"{obj.name} has a vertex weighted to a bone that is not in the original skeleton "
                    f"(vertex {mesh_vertex.index}). Remove that weight or use a bone from the imported skeleton, "
                    "then export again."
                )
            weights.append((joint_index, weight))
    weights.sort(key=itemgetter(1), reverse=True)
    return weights[:12]


def _shape_key_export_data(obj, linear_matrix):
    mesh = obj.data
    shape_keys = getattr(mesh, "shape_keys", None)
    key_blocks = list(getattr(shape_keys, "key_blocks", []) or [])
    if not key_blocks:
        return None, []
    vertex_count = len(mesh.vertices)
    for key in key_blocks:
        if len(key.data) != vertex_count:
            raise ValueError(
                f"Shape key {key.name!r} no longer fits {obj.name}. Delete and recreate that shape key, or "
                "re-import the original model with Import Shape Keys enabled."
            )

    basis_coords = np.empty(vertex_count * 3, dtype=np.float64)
    key_blocks[0].data.foreach_get("co", basis_coords)
    basis_coords = basis_coords.reshape((-1, 3))
    if len(key_blocks) == 1:
        return basis_coords, []

    try:
        metadata = json.loads(str(obj.get("engine_morph_targets_json", "{}") or "{}"))
    except Exception:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    transform = np.array(
        [[float(linear_matrix[row][column]) for column in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    targets = []
    for key in key_blocks[1:]:
        if metadata and str(key.name) not in metadata:
            continue
        key_coords = np.empty(vertex_count * 3, dtype=np.float64)
        key.data.foreach_get("co", key_coords)
        blender_deltas = key_coords.reshape((-1, 3)) - basis_coords
        armature_deltas = blender_deltas @ transform.T
        engine_deltas = np.empty_like(armature_deltas)
        engine_deltas[:, 0] = armature_deltas[:, 0]
        engine_deltas[:, 1] = armature_deltas[:, 2]
        engine_deltas[:, 2] = -armature_deltas[:, 1]

        affected = np.flatnonzero(
            np.linalg.norm(engine_deltas, axis=1) >= MORPH_VERTEX_DELTA_EPSILON
        )
        if not len(affected):
            continue

        stored = metadata.get(str(key.name))
        if isinstance(stored, dict) and stored.get("name"):
            target_name = str(stored["name"])
            target_hash = int(stored.get("hash", string_crc32(target_name))) & U32_MASK
            source_target_index = int(stored.get("index", -1))
        else:
            target_name = str(key.name)
            target_hash = string_crc32(target_name)
            source_target_index = -1
        targets.append({
            "name": target_name,
            "hash": target_hash,
            "source_index": source_target_index,
            "source_deltas": {
                int(index): tuple(float(component) for component in engine_deltas[index])
                for index in affected
            },
        })
    return basis_coords, targets


def _recompute_tangent_for_morph_targets(vertices, indices, targets):
    """Blend a deformed-pose tangent into each morph-affected vertex's static
    packed normal_tangent word.

    Background: the engine's Morph2 runtime only stores/animates *position*
    deltas (see model_morph.py's batch/delta format) - the packed
    normal_tangent word baked per vertex is fixed at bind pose (weight=0) and
    is never touched again. For small, localized Ziva-style targets that
    mismatch is imperceptible; for larger custom blend shapes it shows up as
    warped specular/normal-map shading in the deformed area, since the
    tangent basis no longer matches the actual (deformed) surface.

    This doesn't make the tangent dynamically correct at every blend weight
    (the file format has no room for that - it's one static word), but it
    replaces the pure bind-pose tangent with an average across bind pose and
    each target's fully-applied deformation, which is a strictly better
    single fixed compromise than "always exactly wrong once any target with
    real magnitude is applied at all".
    """
    triangle_count = len(indices) // 3
    if triangle_count == 0:
        return 0

    # Below this displacement, a vertex is only nominally "morph-affected"
    # (present in the delta map with a tiny/near-zero magnitude - common for
    # a target whose mask covers a whole material panel even though most of
    # it barely moves). Recomputing the tangent there anyway introduces a
    # small but real numerical perturbation vs. the original baked tangent
    # (different local triangle-averaging neighborhood), which shows up as a
    # faint, spread-out shading difference across the *entire* panel instead
    # of staying localized to the part that's actually deforming. Vertices
    # below this threshold keep their original bind-pose tangent untouched.
    TANGENT_RECOMPUTE_MIN_DELTA = 0.0002  # meters (0.2mm)

    base_positions = [vertex["co"] for vertex in vertices]
    base_uvs = [vertex.get("uv0") or (0.0, 0.0) for vertex in vertices]

    # tangent_accum[vertex_index] = [sum_x, sum_y, sum_z, flip_sum, contributions]
    tangent_accum = {}

    def accumulate(vertex_index, tangent, flip):
        acc = tangent_accum.get(vertex_index)
        if acc is None:
            acc = [0.0, 0.0, 0.0, 0.0, 0]
            tangent_accum[vertex_index] = acc
        acc[0] += tangent[0]
        acc[1] += tangent[1]
        acc[2] += tangent[2]
        acc[3] += flip
        acc[4] += 1

    for target in targets:
        deltas = target.get("deltas") or {}
        if not deltas:
            continue
        for triangle_index in range(triangle_count):
            i0 = indices[triangle_index * 3]
            i1 = indices[triangle_index * 3 + 1]
            i2 = indices[triangle_index * 3 + 2]
            d0 = deltas.get(i0)
            d1 = deltas.get(i1)
            d2 = deltas.get(i2)
            if d0 is None and d1 is None and d2 is None:
                continue
            p0 = _vec_add(base_positions[i0], d0) if d0 is not None else base_positions[i0]
            p1 = _vec_add(base_positions[i1], d1) if d1 is not None else base_positions[i1]
            p2 = _vec_add(base_positions[i2], d2) if d2 is not None else base_positions[i2]
            uvs = (base_uvs[i0], base_uvs[i1], base_uvs[i2])
            try:
                tangent, flip = _luna_triangle_tangent_space((p0, p1, p2), uvs)
            except Exception:
                continue
            for i, d in ((i0, d0), (i1, d1), (i2, d2)):
                if d is not None and _vec_len(d) >= TANGENT_RECOMPUTE_MIN_DELTA:
                    accumulate(i, tangent, flip)

    if not tangent_accum:
        return 0

    for vertex_index, (sx, sy, sz, flip_sum, count) in tangent_accum.items():
        if count == 0:
            continue
        vertex = vertices[vertex_index]
        bind_tangent = vertex["tangent"]
        # Equal-weight blend between bind pose and the average
        # fully-deformed tangent across every target touching this vertex.
        deformed_avg = (sx / count, sy / count, sz / count)
        blended = _vec_normalize(
            _vec_add(bind_tangent, deformed_avg),
            fallback=bind_tangent,
        )
        # Majority sign of the flips seen across contributing targets; tie
        # (flip_sum == 0) keeps the original bind-pose handedness.
        blended_flip = 1.0 if flip_sum >= 0 else -1.0

        normal_tangent, tangent_y = _pack_normal_tangent(vertex["normal"], blended)
        old_position_w = int(vertex["position_w"])
        extrusion_encoded = (abs(old_position_w) >> 10) & 0x1F
        vertex["normal_tangent"] = normal_tangent
        vertex["position_w"] = _pack_position_w(tangent_y, blended_flip, extrusion_encoded)
        vertex["tangent"] = blended

    return len(tangent_accum)


def _finalize_export_morph_topology(vertices, indices, source_targets):
    targets = []
    affected_vertices = set()
    for target in source_targets:
        source_deltas = target["source_deltas"]
        deltas = {}
        for export_index, vertex in enumerate(vertices):
            delta = source_deltas.get(int(vertex["source_index"]))
            if delta is not None:
                deltas[export_index] = delta
        if deltas:
            targets.append({
                "name": target["name"],
                "hash": target["hash"],
                "source_index": target.get("source_index", -1),
                "deltas": deltas,
            })
            affected_vertices.update(deltas)
    if not affected_vertices:
        return vertices, indices, targets, 0

    order = sorted(affected_vertices) + [index for index in range(len(vertices)) if index not in affected_vertices]
    remap = {old_index: new_index for new_index, old_index in enumerate(order)}
    reordered_vertices = [vertices[old_index] for old_index in order]
    reordered_indices = [remap[int(index)] for index in indices]
    for target in targets:
        target["deltas"] = {remap[index]: delta for index, delta in target["deltas"].items()}
    return reordered_vertices, reordered_indices, targets, len(affected_vertices)


def _order_export_vertices_by_control_point(vertices, indices, control_vertex_count):
  
    primary = {}
    duplicates = []
    for old_index, vertex in enumerate(vertices):
        source_index = int(vertex.get("source_index", -1))
        if 0 <= source_index < int(control_vertex_count) and source_index not in primary:
            primary[source_index] = old_index
        else:
            duplicates.append(old_index)
    if len(primary) != int(control_vertex_count):
        # Preserve established behavior for malformed meshes containing unused
        # control points; they cannot safely satisfy direct Luna vertex IDs.
        return vertices, indices
    order = [primary[index] for index in range(int(control_vertex_count))]
    order.extend(sorted(duplicates, key=lambda index: (int(vertices[index].get("source_index", -1)), index)))
    if order == list(range(len(vertices))):
        return vertices, indices
    remap = {old_index: new_index for new_index, old_index in enumerate(order)}
    return [vertices[index] for index in order], [remap[int(index)] for index in indices]


def _prefetch_loop_uvs(layer, loop_count):
    if not layer:
        return [(0.0, 0.0)] * loop_count
    flat = np.empty(loop_count * 2, dtype=np.float64)
    layer.data.foreach_get("uv", flat)
    return list(zip(flat[0::2].tolist(), (1.0 - flat[1::2]).tolist()))


def _uv_nearly_equal(left, right, threshold=0.0001):
    if left is None or right is None:
        return left is None and right is None
    return (
        abs(float(left[0]) - float(right[0])) <= threshold
        and abs(float(left[1]) - float(right[1])) <= threshold
    )


def _prefetch_engine_positions(mesh, matrix, basis_coords, vertex_count):
    if _matrix_is_identity_4x4(matrix):
        if basis_coords is not None:
            coords = np.asarray(basis_coords, dtype=np.float32).astype(np.float64)
        else:
            raw = np.empty(vertex_count * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", raw)
            coords = raw.reshape(-1, 3)
        return list(zip(
            coords[:, 0].tolist(),
            coords[:, 2].tolist(),
            (-coords[:, 1]).tolist(),
        ))
    if basis_coords is not None:
        return [
            _blender_to_engine_vec(matrix @ mathutils.Vector(basis_coords[index]))
            for index in range(vertex_count)
        ]
    return [_blender_to_engine_vec(matrix @ vertex.co) for vertex in mesh.vertices]


def _matrix_is_identity_4x4(matrix):
    try:
        for row in range(4):
            for column in range(4):
                expected = 1.0 if row == column else 0.0
                if float(matrix[row][column]) != expected:
                    return False
    except Exception:
        return False
    return True


def _normalize_engine_rows(rows):
    x = rows[:, 0]
    y = rows[:, 1]
    z = rows[:, 2]
    length = np.sqrt(np.maximum(0.0, x * x + y * y + z * z))
    usable = length > 1.0e-8
    safe = np.where(usable, length, 1.0)
    xs = (x / safe).tolist()
    ys = (y / safe).tolist()
    zs = (z / safe).tolist()
    flags = usable.tolist()
    return [
        (xs[index], ys[index], zs[index]) if flags[index] else None
        for index in range(len(flags))
    ]


def _engine_frame_rows(decoded, transform, transform_is_identity):
    if transform_is_identity:
        narrowed = decoded.astype(np.float32).astype(np.float64)
        return np.column_stack((narrowed[:, 0], narrowed[:, 1], narrowed[:, 2]))
    rows = np.empty_like(decoded)
    for index in range(len(decoded)):
        blender_vector = mathutils.Vector((
            float(decoded[index][0]),
            -float(decoded[index][2]),
            float(decoded[index][1]),
        ))
        rows[index] = _blender_to_engine_vec(transform @ blender_vector)
    return rows


def _prefetch_source_basis_frames(
    source_basis,
    vertex_count,
    normal_matrix,
    linear_matrix,
    normal_matrix_identity,
    linear_matrix_identity,
    tangent_flip_transform,
):
    words_signed = np.empty(vertex_count, dtype=np.int32)
    source_basis["engine_source_normal_tangent"].foreach_get("value", words_signed)
    position_w = np.empty(vertex_count, dtype=np.int32)
    source_basis["engine_position_w"].foreach_get("value", position_w)

    words = words_signed.astype(np.uint32)
    normals = _engine_frame_rows(
        _decode_packed_normal_array(words), normal_matrix, normal_matrix_identity
    )
    tangents = _engine_frame_rows(
        _decode_packed_tangent_array(words, position_w), linear_matrix, linear_matrix_identity
    )
    flips = np.where(position_w >= 0, 1.0, -1.0) * tangent_flip_transform
    return (
        words.tolist(),
        position_w.tolist(),
        _normalize_engine_rows(normals),
        _normalize_engine_rows(tangents),
        flips.tolist(),
    )


def _export_mesh_vertices(
    obj,
    arm,
    source_joint_count,
    original_flags=0,
    export_warnings=None,
    fallback_morph_targets=None,
):
    mesh = obj.data
    uv_info = _ensure_model_uv_layers(obj, export_warnings)
    mesh.calc_loop_triangles()
    uv0_layer = _uv_layer_by_name_or_index(mesh, "UV0", 0)
    uv1_layer = _uv_layer_by_name_or_index(mesh, "UV1", 1)
    uv2_layer = _uv_layer_by_name_or_index(mesh, "UV2", 2)
    has_uv1 = _should_export_uv_channel(obj, uv_info, 1, original_flags)
    has_uv2 = _should_export_uv_channel(obj, uv_info, 2, original_flags)
    matrix = _safe_matrix_relative_to_armature(arm, obj)
    linear_matrix = matrix.to_3x3()
    try:
        normal_matrix = linear_matrix.inverted().transposed()
    except Exception:
        normal_matrix = linear_matrix
    group_to_joint = _vertex_group_joint_map(obj, arm)
    basis_coords, source_morph_targets = _shape_key_export_data(obj, linear_matrix)
    if not source_morph_targets and fallback_morph_targets:
        source_morph_targets = [
            {
                "name": str(target["name"]),
                "hash": int(target["hash"]) & U32_MASK,
                "source_index": int(target.get("source_index", target.get("index", -1))),
                "source_deltas": {
                    int(index): tuple(delta)
                    for index, delta in target.get("deltas", {}).items()
                },
            }
            for target in fallback_morph_targets
        ]

    if not uv0_layer:
        raise ValueError(
            f"{obj.name} needs a UV map called UV0. In Object Data Properties > UV Maps, create or rename "
            "the main texture UV map to UV0, then export again."
        )
    uv0_layer = _uv_layer_by_name_or_index(mesh, "UV0", 0)
    uv1_layer = _uv_layer_by_name_or_index(mesh, "UV1", 1)
    uv2_layer = _uv_layer_by_name_or_index(mesh, "UV2", 2)

    source_position_w = None
    try:
        attr = mesh.attributes.get("engine_position_w")
        if attr and attr.domain == 'POINT' and len(attr.data) == len(mesh.vertices):
            source_position_w = attr.data
    except Exception:
        source_position_w = None

    msmr_tangent = mesh.attributes.get("MSMR_Tangent")
    msmr_tangent_sign = mesh.attributes.get("MSMR_BitangentSign")
    msmr_tangent_valid = mesh.attributes.get("MSMR_TangentValid")
    if not (
        msmr_tangent is not None
        and msmr_tangent.domain == 'POINT'
        and len(msmr_tangent.data) == len(mesh.vertices)
    ):
        msmr_tangent = None
    if not (
        msmr_tangent_sign is not None
        and msmr_tangent_sign.domain == 'POINT'
        and len(msmr_tangent_sign.data) == len(mesh.vertices)
    ):
        msmr_tangent_sign = None
    if not (
        msmr_tangent_valid is not None
        and msmr_tangent_valid.domain == 'POINT'
        and len(msmr_tangent_valid.data) == len(mesh.vertices)
    ):
        msmr_tangent_valid = None

    source_basis = _source_model_basis(mesh, uv0_layer, basis_coords)
    source_basis_packed_exact = bool(source_basis and _linear_matrix_is_identity(linear_matrix))
    try:
        tangent_flip_transform = -1.0 if float(linear_matrix.determinant()) < 0.0 else 1.0
    except Exception:
        tangent_flip_transform = 1.0

    export_vertices = []
    match_keys = []
    vertex_buckets = {}
    indices = []

    vertex_count = len(mesh.vertices)
    loop_count = len(mesh.loops)
    triangle_count = len(mesh.loop_triangles)

    loop_vertex_index = np.empty(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_index)
    loop_vertex_index = loop_vertex_index.tolist()

    triangle_loop_indices = np.empty(triangle_count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", triangle_loop_indices)
    triangle_loop_indices = triangle_loop_indices.tolist()

    uv0_values = _prefetch_loop_uvs(uv0_layer, loop_count)
    uv1_values = _prefetch_loop_uvs(uv1_layer, loop_count) if has_uv1 and uv1_layer else None
    uv2_values = _prefetch_loop_uvs(uv2_layer, loop_count) if has_uv2 and uv2_layer else None

    positions_engine = _prefetch_engine_positions(mesh, matrix, basis_coords, vertex_count)

    normal_matrix_identity = _linear_matrix_is_identity(normal_matrix, threshold=0.0)
    corner_normals_raw = []

    source_extrusion = None
    if source_position_w is not None:
        extrusion_values = np.empty(vertex_count, dtype=np.int32)
        source_position_w.foreach_get("value", extrusion_values)
        source_extrusion = ((np.abs(extrusion_values.astype(np.int64)) >> 10) & 0x1F).tolist()

    source_words = source_ws = source_normals = source_tangents = source_flips = None
    if source_basis is not None:
        (
            source_words,
            source_ws,
            source_normals,
            source_tangents,
            source_flips,
        ) = _prefetch_source_basis_frames(
            source_basis,
            vertex_count,
            normal_matrix,
            linear_matrix,
            normal_matrix_identity,
            _linear_matrix_is_identity(linear_matrix, threshold=0.0),
            tangent_flip_transform,
        )

    weights_cache = {}

    def corner_normal_engine(loop_index):
        if not corner_normals_raw:
            values = np.empty(loop_count * 3, dtype=np.float64)
            mesh.corner_normals.foreach_get("vector", values)
            corner_normals_raw.extend(values.tolist())
        base = loop_index * 3
        raw_x = corner_normals_raw[base]
        raw_y = corner_normals_raw[base + 1]
        raw_z = corner_normals_raw[base + 2]
        if normal_matrix_identity:
            return _vec_normalize((raw_x, raw_z, -raw_y))
        return _vec_normalize(
            _blender_to_engine_vec(normal_matrix @ mathutils.Vector((raw_x, raw_y, raw_z)))
        )

    for triangle_index in range(triangle_count):
        base = triangle_index * 3
        triangle_loops = (
            triangle_loop_indices[base],
            triangle_loop_indices[base + 1],
            triangle_loop_indices[base + 2],
        )
        triangle_vertices = (
            loop_vertex_index[triangle_loops[0]],
            loop_vertex_index[triangle_loops[1]],
            loop_vertex_index[triangle_loops[2]],
        )
        triangle_positions = (
            positions_engine[triangle_vertices[0]],
            positions_engine[triangle_vertices[1]],
            positions_engine[triangle_vertices[2]],
        )
        triangle_basis = None

        tri_indices = []
        for triangle_corner in range(3):
            loop_index = triangle_loops[triangle_corner]
            source_vertex_index = triangle_vertices[triangle_corner]
            uv0 = uv0_values[loop_index]
            uv1 = uv1_values[loop_index] if uv1_values is not None else None
            uv2 = uv2_values[loop_index] if uv2_values is not None else None
            packed_source_basis = None
            if source_basis is not None:
                normal = source_normals[source_vertex_index]
                if normal is None:
                    normal = corner_normal_engine(loop_index)
                tangent = source_tangents[source_vertex_index]
                if tangent is None:
                    if triangle_basis is None:
                        triangle_basis = _luna_triangle_tangent_space(
                            triangle_positions,
                            (
                                uv0_values[triangle_loops[0]],
                                uv0_values[triangle_loops[1]],
                                uv0_values[triangle_loops[2]],
                            ),
                        )
                    tangent = triangle_basis[0]
                tangent_flip = source_flips[source_vertex_index]
                if source_basis_packed_exact:
                    packed_source_basis = (
                        source_words[source_vertex_index],
                        source_ws[source_vertex_index],
                    )
            else:
                normal = corner_normal_engine(loop_index)
                if triangle_basis is None:
                    triangle_basis = _luna_triangle_tangent_space(
                        triangle_positions,
                        (
                            uv0_values[triangle_loops[0]],
                            uv0_values[triangle_loops[1]],
                            uv0_values[triangle_loops[2]],
                        ),
                    )
                tangent, tangent_flip = triangle_basis
            tangent_attribute_valid = bool(
                msmr_tangent is not None
                and (
                    msmr_tangent_valid is None
                    or msmr_tangent_valid.data[source_vertex_index].value
                )
            )
            if tangent_attribute_valid:
                tangent_vector = msmr_tangent.data[source_vertex_index].vector
                tangent_engine = _vec_normalize(_blender_to_engine_vec(
                    linear_matrix @ mathutils.Vector(tangent_vector)
                ))
                tangent_sign = (
                    float(msmr_tangent_sign.data[source_vertex_index].value)
                    if msmr_tangent_sign is not None
                    else -float(tangent_flip)
                )
                attribute_flip = -tangent_sign * tangent_flip_transform
                tangent_was_edited = (
                    # Decoding and re-exposing the packed source tangent as a
                    # float attribute introduces a few ulps around zero. Keep
                    # the exact packed word unless the authored direction moved
                    # by more than that round-trip noise.
                    _vec_dot(tangent_engine, tangent) < 0.9999
                    or (attribute_flip >= 0.0) != (float(tangent_flip) >= 0.0)
                )
                if source_basis is None or tangent_was_edited:
                    tangent = tangent_engine
                    tangent_flip = attribute_flip
                    packed_source_basis = None
            extrusion_encoded = 16
            if source_extrusion is not None:
                extrusion_encoded = source_extrusion[source_vertex_index]
            export_index = None
            flip_positive = tangent_flip >= 0.0
            normal_x, normal_y, normal_z = normal
            tangent_x, tangent_y, tangent_z = tangent
            uv0_u, uv0_v = uv0
            for candidate_index in vertex_buckets.get(source_vertex_index, ()):
                key = match_keys[candidate_index]
                other = key[0]
                if (
                    other[0] * normal_x + other[1] * normal_y + other[2] * normal_z
                    < VERTEX_MATCH_NORMAL_DOT_MIN
                ):
                    continue
                other = key[1]
                if (
                    other[0] * tangent_x + other[1] * tangent_y + other[2] * tangent_z
                    < VERTEX_MATCH_TANGENT_DOT_MIN
                ):
                    continue
                if key[2] != flip_positive:
                    continue
                other = key[3]
                if (
                    abs(other[0] - uv0_u) > VERTEX_MATCH_UV_THRESHOLD
                    or abs(other[1] - uv0_v) > VERTEX_MATCH_UV_THRESHOLD
                ):
                    continue
                if uv1 is not None:
                    other = key[4]
                    if (
                        abs(other[0] - uv1[0]) > VERTEX_MATCH_UV_THRESHOLD
                        or abs(other[1] - uv1[1]) > VERTEX_MATCH_UV_THRESHOLD
                    ):
                        continue
                if uv2 is not None:
                    other = key[5]
                    if (
                        abs(other[0] - uv2[0]) > VERTEX_MATCH_UV_THRESHOLD
                        or abs(other[1] - uv2[1]) > VERTEX_MATCH_UV_THRESHOLD
                    ):
                        continue
                export_index = candidate_index
                candidate = export_vertices[candidate_index]
                candidate["_normal_sum"] = _vec_add(candidate["_normal_sum"], normal)
                candidate["_tangent_sum"] = _vec_add(candidate["_tangent_sum"], tangent)
                candidate["_basis_count"] += 1
                break
            if export_index is None:
                weights = weights_cache.get(source_vertex_index)
                if weights is None:
                    weights = _vertex_weights(
                        mesh.vertices[source_vertex_index],
                        obj,
                        group_to_joint,
                        source_joint_count,
                    )
                    weights_cache[source_vertex_index] = weights
                export_index = len(export_vertices)
                vertex_buckets.setdefault(source_vertex_index, []).append(export_index)
                match_keys.append((normal, tangent, flip_positive, uv0, uv1, uv2))
                export_vertices.append({
                    "source_index": source_vertex_index,
                    "co": triangle_positions[triangle_corner],
                    "normal": normal,
                    "tangent": tangent,
                    "tangent_flip": tangent_flip,
                    "extrusion_encoded": extrusion_encoded,
                    "_normal_sum": normal,
                    "_tangent_sum": tangent,
                    "_basis_count": 1,
                    "_packed_source_basis": packed_source_basis,
                    "uv0": uv0,
                    "uv1": uv1,
                    "uv2": uv2,
                    "weights": list(weights),
                })
            tri_indices.append(export_index)
        indices.extend(tri_indices)


    for vertex in export_vertices:
        normal = _vec_normalize(vertex.pop("_normal_sum"))
        tangent = _vec_normalize(vertex.pop("_tangent_sum"), fallback=(1.0, 0.0, 0.0))
        vertex.pop("_basis_count", None)
        vertex["normal"] = normal
        vertex["tangent"] = tangent
        packed_source_basis = vertex.pop("_packed_source_basis", None)
        tangent_flip = vertex.pop("tangent_flip")
        extrusion_encoded = vertex.pop("extrusion_encoded")
        if packed_source_basis is not None:
            vertex["normal_tangent"] = int(packed_source_basis[0]) & U32_MASK
            vertex["position_w"] = int(packed_source_basis[1])
        else:
            normal_tangent, tangent_y = _pack_normal_tangent(normal, tangent)
            vertex["normal_tangent"] = normal_tangent
            vertex["position_w"] = _pack_position_w(
                tangent_y,
                tangent_flip,
                extrusion_encoded,
            )

    export_vertices, indices = _order_export_vertices_by_control_point(
        export_vertices,
        indices,
        len(mesh.vertices),
    )

    export_vertices, indices, morph_targets, anim_vert_count = _finalize_export_morph_topology(
        export_vertices,
        indices,
        source_morph_targets,
    )
    if morph_targets:
        _recompute_tangent_for_morph_targets(export_vertices, indices, morph_targets)
    return export_vertices, indices, bool(has_uv1), bool(has_uv2), morph_targets, anim_vert_count


def _split_export_vertex_chunks(vertices, indices, max_vertices=MODEL_SPLIT_VERTEX_TARGET):
    chunks = []
    chunk_vertices = []
    chunk_indices = []
    remap = {}
    max_vertices = max(3, min(int(max_vertices), MODEL_MAX_VERTEX_COUNT))

    triangles = []
    for tri_order, tri_start in enumerate(range(0, len(indices), 3)):
        tri = indices[tri_start:tri_start + 3]
        if len(tri) < 3:
            continue
        coords = [vertices[int(index)]["co"] for index in tri]
        centroid = (
            (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0,
            (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0,
            (coords[0][2] + coords[1][2] + coords[2][2]) / 3.0,
        )
        triangles.append((-centroid[1], centroid[0], centroid[2], tri_order, tri))
    triangles.sort()

    def flush_chunk():
        nonlocal chunk_vertices, chunk_indices, remap
        if chunk_vertices and chunk_indices:
            chunks.append((chunk_vertices, chunk_indices))
        chunk_vertices = []
        chunk_indices = []
        remap = {}

    for _height_key, _x_key, _z_key, _tri_order, tri in triangles:
        missing = []
        for source_index in tri:
            if source_index not in remap and source_index not in missing:
                missing.append(source_index)
        if chunk_vertices and len(chunk_vertices) + len(missing) > max_vertices:
            flush_chunk()

        for source_index in tri:
            mapped = remap.get(source_index)
            if mapped is None:
                mapped = len(chunk_vertices)
                remap[source_index] = mapped
                chunk_vertices.append(vertices[source_index])
            chunk_indices.append(mapped)

        if len(chunk_vertices) > max_vertices:
            raise ValueError(
                "One face is too large for the game format. Apply the mesh modifiers, triangulate the mesh, "
                "and split very large geometry into smaller objects before exporting again."
            )

    flush_chunk()
    return chunks


def _uv_log_for_values(vertices, channel_name):
    max_abs = 1.0
    for vertex in vertices:
        uv = vertex.get(channel_name)
        if uv is None:
            continue
        max_abs = max(max_abs, abs(float(uv[0])), abs(float(uv[1])))
    unbounded = int(round(max_abs * MODEL_UV_FLOAT_TO_FIXED_BASE)) >> 15
    return _clamp(unbounded.bit_length() if unbounded > 0 else 0, 0, 15)


def _uv_scale(log_value):
    return float(1 << int(log_value)) / MODEL_UV_FLOAT_TO_FIXED_BASE


def _pack_uv(uv, log_value):
    scale = _uv_scale(log_value)
    return (
        _clamp_i16(float(uv[0]) / scale),
        _clamp_i16(float(uv[1]) / scale),
    )


def _encode_azimuthal(vector):
    vector = _vec_normalize(vector)
    inv_f = 1.0 / math.sqrt(abs(vector[2]) * 4.0 + 4.0)
    return (
        vector[0] * inv_f + 0.5,
        vector[1] * inv_f + 0.5,
        0.0 if vector[2] < 0.0 else 1.0,
    )


def _pack_normal_tangent(normal, tangent):
    normal_azim = _encode_azimuthal(normal)
    tangent_azim = _encode_azimuthal(tangent)
    norm_tan_z = normal_azim[2] * (2.0 / 3.0) + tangent_azim[2] * (1.0 / 3.0)
    nx = _clamp(_round_engine(normal_azim[0] * 1023.0), 0, 1023)
    ny = _clamp(_round_engine(normal_azim[1] * 1023.0), 0, 1023)
    tx = _clamp(_round_engine(tangent_azim[0] * 1023.0), 0, 1023)
    ty = _clamp(_round_engine(tangent_azim[1] * 1023.0), 0, 1023)
    nz_tz = _clamp(_round_engine(norm_tan_z * 3.0), 0, 3)
    packed = int(nx) | (int(ny) << 10) | (int(tx) << 20) | (int(nz_tz) << 30)
    return packed, int(ty)


def _packed_normal_key(normal):
    normal_azim = _encode_azimuthal(normal)
    return (
        _clamp(_round_engine(normal_azim[0] * 1023.0), 0, 1023),
        _clamp(_round_engine(normal_azim[1] * 1023.0), 0, 1023),
        normal_azim[2] >= 0.5,
    )


def _pack_position_w(tangent_y, tangent_flip, extrusion_encoded=16):
    magnitude = (int(tangent_y) & 0x3FF) | ((int(extrusion_encoded) & 0x1F) << 10)
    if float(tangent_flip) >= 0.0:
        return magnitude
    return -1 if magnitude == 0 else -magnitude


def _normalize_skin_weights(weights):
    if not weights:
        return [(0, 256)]
    weights = weights[:12]
    total = sum(max(0.0, weight) for _joint, weight in weights)
    if total <= 1e-8:
        return [(int(weights[0][0]), 256)]
    scaled = []
    running = 0
    for joint, weight in weights:
        exact = max(0.0, weight) * 256.0 / total
        base = int(math.floor(exact))
        scaled.append([int(joint), base, exact - base])
        running += base
    remainder = 256 - running
    scaled.sort(key=itemgetter(2), reverse=True)
    for index in range(abs(remainder)):
        scaled[index % len(scaled)][1] += 1 if remainder > 0 else -1
    scaled.sort(key=itemgetter(1), reverse=True)
    result = [(joint, _clamp(weight, 0, 256)) for joint, weight, _frac in scaled if weight > 0]
    return result or [(int(weights[0][0]), 256)]


def _skin_joint_entries(weights):
    entries = [(int(joint), int(weight)) for joint, weight in weights if int(weight) > 0]
    return entries or [(int(weights[0][0]) if weights else 0, 256)]


def _split_full_influence_joints(joints, is_16bit, cluster_joint_count):
    if cluster_joint_count > 1 and len(joints) >= 1 and joints[0][1] == 256:
        joint_index = joints[0][0]
        joints[0] = (joint_index, 128)
        joints.insert(1, (joint_index if is_16bit else 0, 128))
        return True
    return False


def _insert_bridge_indices(joints, joint_count_max):
    joint_count = len(joints)
    index = 0
    while index < joint_count:
        joint_index, weight = joints[index]
        if joint_index <= SKIN_UINT8_MAX:
            index += 1
            continue
        joints.insert(index + 1, (joint_index - SKIN_UINT8_MAX, weight))
        joints[index] = (SKIN_UINT8_MAX, 0)
        joint_count += 1
        if joint_count >= joint_count_max:
            break
        index += 1


def _prepare_cluster_skin(normalized_weights):
    cluster_vertex_count = len(normalized_weights)
    source_joints = [_skin_joint_entries(weights) for weights in normalized_weights]

    joint_min = None
    joint_count_max = 1
    for joints in source_joints:
        for joint_index, weight in joints:
            if weight > 0:
                joint_min = joint_index if joint_min is None else min(joint_min, joint_index)
        joint_count_max = max(joint_count_max, len(joints))
    if joint_min is None:
        joint_min = 0
    joint_offset = min((joint_min // SKIN_JOINT_OFFSET_STEP) * SKIN_JOINT_OFFSET_STEP, SKIN_JOINT_OFFSET_MAX)
    if joint_count_max == 0 or joint_count_max > 12:
        raise ValueError(
            "Some vertices use more than 12 bone weights. In Weight Paint mode, limit each vertex to 12 "
            "bones or fewer, normalize the weights, then export again."
        )

    prepared = []
    is_16bit = False
    for joints in source_joints:
        working = [(joint_index - joint_offset, weight) for joint_index, weight in joints]
        working.sort(key=lambda item: item[0])
        if working[0][0] > SKIN_UINT8_MAX:
            is_16bit = True
            break

        incremental = list(working)
        for index in range(len(incremental) - 1, 0, -1):
            incremental[index] = (incremental[index][0] - incremental[index - 1][0], incremental[index][1])

        if len(incremental) < joint_count_max:
            _insert_bridge_indices(incremental, joint_count_max)

        if any(joint_index > SKIN_UINT8_MAX for joint_index, _weight in incremental):
            is_16bit = True
            break
        prepared.append(incremental)

    if is_16bit:
        joint_offset = 0
        prepared = []
        for joints in source_joints:
            absolute = sorted(joints, key=lambda item: item[0])
            prepared.append(list(absolute))
        if joint_count_max > 2:
            joint_count_max = min(12, _align(joint_count_max, 4))

    for index, joints in enumerate(prepared):
        _split_full_influence_joints(joints, is_16bit, joint_count_max)
        while len(joints) < joint_count_max:
            joints.append((0, 0))
        prepared[index] = joints[:joint_count_max]

    return is_16bit, joint_offset, joint_count_max, prepared[:cluster_vertex_count]


def _serialize_cluster_skin_data(prepared, is_16bit, joint_count_max):
    cluster_bytes = bytearray()
    for joints in prepared:
        for joint_index, weight in joints[:joint_count_max]:
            if is_16bit:
                cluster_bytes += struct.pack("<H", int(joint_index) & 0xFFFF)
                if joint_count_max > 1:
                    cluster_bytes += struct.pack("<B", int(weight) & 0xFF)
            else:
                cluster_bytes += struct.pack("<B", int(joint_index) & 0xFF)
                if joint_count_max > 1:
                    cluster_bytes += struct.pack("<B", int(weight) & 0xFF)
    pad = (-len(cluster_bytes)) & 3
    if pad:
        cluster_bytes += b"\x00" * pad
    return bytes(cluster_bytes)


def _build_skin_sections(vertices, force_skin, anim_cluster_count=0):
    if not force_skin:
        return b"", b""
    skin_data = bytearray()
    cluster_headers = bytearray()
    for cluster_index, cluster_start in enumerate(range(0, len(vertices), SKIN_CLUSTER_VERTEX_COUNT)):
        cluster_vertices = vertices[cluster_start:cluster_start + SKIN_CLUSTER_VERTEX_COUNT]
        normalized_weights = [_normalize_skin_weights(vertex.get("weights", [])) for vertex in cluster_vertices]
        influence_count = max(1, min(12, max(len(weights) for weights in normalized_weights)))

        if influence_count > 1:
            normalized_weights = [
                [(weights[0][0], 128), (weights[0][0], 128)]
                if len(weights) == 1 and weights[0][1] == 256 else weights
                for weights in normalized_weights
            ]
            influence_count = max(2, min(12, max(len(weights) for weights in normalized_weights)))

        is_16bit, joint_offset, joint_count_max, prepared = _prepare_cluster_skin(normalized_weights)

        _align_buffer(skin_data, SKIN_CLUSTER_WORD_BYTES)
        data_offset4 = len(skin_data) // SKIN_CLUSTER_WORD_BYTES
        if data_offset4 > SKIN_CLUSTER_OFFSET_MASK:
            raise ValueError(
                "This weighted mesh is too large for one game mesh part. Split it into smaller objects, keep "
                "their Armature parent and weights, then export again."
            )
        header = data_offset4 | ((joint_count_max - 1) << SKIN_CLUSTER_INFLUENCE_SHIFT)
        if is_16bit:
            header |= SKIN_CLUSTER_FULL_INDEX_BIT
        else:
            header |= (joint_offset // SKIN_JOINT_OFFSET_STEP) << SKIN_CLUSTER_JOINT_OFFSET_SHIFT
        if cluster_index < int(anim_cluster_count):
            header |= SKIN_CLUSTER_ANIM_VERT_BIT
        cluster_headers += struct.pack("<I", header)
        skin_data += _serialize_cluster_skin_data(prepared, is_16bit, joint_count_max)
    return bytes(skin_data), bytes(cluster_headers)


def _triangle_area(a, b, c):
    return 0.5 * _vec_len(_vec_cross(_vec_sub(b, a), _vec_sub(c, a)))


def _triangle_uv_area(a, b, c):
    return abs(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    ) * 0.5


def _repair_missing_skin_weights(vertices, indices):
    missing = {index for index, vertex in enumerate(vertices) if not vertex.get("weights")}
    if not missing:
        return 0, 0

    source_vertices = {
        int(vertices[index].get("source_index", index))
        for index in missing
    }
    adjacency = [set() for _vertex in vertices]
    for index in range(0, len(indices) - 2, 3):
        a, b, c = (int(indices[index]), int(indices[index + 1]), int(indices[index + 2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    while missing:
        updates = {}
        for vertex_index in missing:
            neighbor_weights = [
                vertices[neighbor].get("weights", [])
                for neighbor in adjacency[vertex_index]
                if vertices[neighbor].get("weights")
            ]
            if not neighbor_weights:
                continue
            totals = {}
            for weights in neighbor_weights:
                for joint, weight in weights:
                    totals[int(joint)] = totals.get(int(joint), 0.0) + float(weight)
            divisor = float(len(neighbor_weights))
            updates[vertex_index] = sorted(
                ((joint, weight / divisor) for joint, weight in totals.items() if weight > 0.0),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        if not updates:
            break
        for vertex_index, weights in updates.items():
            vertices[vertex_index]["weights"] = weights
        missing.difference_update(updates)

    fallback_count = len(missing)
    if missing:
        joint_totals = {}
        for vertex in vertices:
            for joint, weight in vertex.get("weights", []):
                joint_totals[int(joint)] = joint_totals.get(int(joint), 0.0) + float(weight)
        fallback_joint = max(joint_totals, key=joint_totals.get) if joint_totals else 0
        for vertex_index in missing:
            vertices[vertex_index]["weights"] = [(fallback_joint, 1.0)]

    return len(source_vertices), fallback_count


def _fit_subset_mpu(vertices, requested_mpu):
    coords = [vertex["co"] for vertex in vertices]
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    extents = [(maxs[axis] - mins[axis]) * 0.5 for axis in range(3)]
    radius = max((_vec_len(_vec_sub(coord, center)) for coord in coords), default=0.0)
    residual_limit = 32767 - (1 << (SUBSET_CENTER_LOG_SCALE - 1))
    required_mpu = max(float(requested_mpu), 1e-12)
    for axis in range(3):
        required_mpu = max(
            required_mpu,
            extents[axis] / float(residual_limit),
            abs(center[axis]) / float(32767 * (1 << SUBSET_CENTER_LOG_SCALE)),
            extents[axis] / float(255 * (1 << SUBSET_CENTER_LOG_SCALE)),
        )
    required_mpu = max(
        required_mpu,
        radius / float(255 * (2 << SUBSET_CENTER_LOG_SCALE)),
    )
    return math.nextafter(required_mpu, math.inf)


def _subset_bounds(vertices, mpu):
    coords = [vertex["co"] for vertex in vertices]
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    extents = tuple((maxs[axis] - mins[axis]) * 0.5 for axis in range(3))
    radius = max((_vec_len(_vec_sub(coord, center)) for coord in coords), default=0.0)

    units = [
        [int(round(coord[axis] / mpu)) for coord in coords]
        for axis in range(3)
    ]
    needs_origin = any(
        min(units[axis]) < -32768 or max(units[axis]) > 32767
        for axis in range(3)
    )
    origin_packed = [
        _clamp_i16(center[axis] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE)))
        for axis in range(3)
    ]
    origin_units = [
        packed << SUBSET_CENTER_LOG_SCALE if needs_origin else 0
        for packed in origin_packed
    ]

    extents_packed = (
        int(_clamp(math.ceil(extents[0] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(extents[1] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(extents[2] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(radius / (mpu * float(2 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
    )
    extents_word = (
        extents_packed[0]
        | (extents_packed[1] << 8)
        | (extents_packed[2] << 16)
        | (extents_packed[3] << 24)
    )
    return center, extents, radius, origin_units, origin_packed, extents_word, needs_origin


def _build_subset_geometry_from_data(
    obj,
    arm,
    material_index,
    original_record,
    custom_stream_index,
    vertices,
    indices,
    has_uv1,
    has_uv2,
    anim_vert_count=0,
    export_warnings=None,
):
    original_flags = 0
    if original_record:
        original_flags = struct.unpack_from("<H", original_record, MODEL_SUBSET_FLAGS_OFFSET)[0]
    if len(vertices) > MODEL_MAX_VERTEX_COUNT:
        raise ValueError(
            f"{obj.name} is too large for one game mesh part. Split it into smaller meshes with fewer than "
            f"{MODEL_MAX_VERTEX_COUNT} export vertices each, then try again."
        )
    if any(index >= MODEL_MAX_VERTEX_COUNT for index in indices):
        raise ValueError(
            f"{obj.name} is too large for one game mesh part. Split it into smaller meshes, then try again."
        )

    mpu = float(obj.get("engine_mpu", arm.get("engine_mpu", MODEL_DEFAULT_MPU) if arm else MODEL_DEFAULT_MPU) or MODEL_DEFAULT_MPU)
    if mpu <= 0.0:
        mpu = MODEL_DEFAULT_MPU
    mpu = _fit_subset_mpu(vertices, mpu)

    uv0_log = _uv_log_for_values(vertices, "uv0")
    uv1_log = _uv_log_for_values(vertices, "uv1") if has_uv1 else 0
    uv2_log = _uv_log_for_values(vertices, "uv2") if has_uv2 else 0
    uv_log_scales = int(uv0_log) | (int(uv1_log) << 4) | (int(uv2_log) << 8)

    center, extents, radius, origin_units, origin_packed, extents_word, needs_origin = _subset_bounds(vertices, mpu)

    pack_std_vertex = struct.Struct("<hhhhIhh").pack
    origin_x, origin_y, origin_z = origin_units[0], origin_units[1], origin_units[2]
    uv0_scale = _uv_scale(uv0_log)
    std_vertex_records = []
    append_std_vertex = std_vertex_records.append
    for vertex in vertices:
        co = vertex["co"]
        uv0 = vertex.get("uv0", (0.0, 0.0))
        append_std_vertex(pack_std_vertex(
            _clamp(int(round(co[0] / mpu)) - origin_x, -32768, 32767),
            _clamp(int(round(co[1] / mpu)) - origin_y, -32768, 32767),
            _clamp(int(round(co[2] / mpu)) - origin_z, -32768, 32767),
            int(vertex["position_w"]),
            int(vertex["normal_tangent"]),
            _clamp_i16(float(uv0[0]) / uv0_scale),
            _clamp_i16(float(uv0[1]) / uv0_scale),
        ))
    std_vertices = b"".join(std_vertex_records)

    geom = bytearray()
    vertex_std_offset = 0
    geom += std_vertices

    vertex_uv12_offset = 0
    if has_uv1 or has_uv2:
        _align_buffer(geom, 4)
        vertex_uv12_offset = len(geom)
        for vertex in vertices:
            if has_uv1:
                geom += struct.pack("<hh", *_pack_uv(vertex.get("uv1") or (0.0, 0.0), uv1_log))
            if has_uv2:
                geom += struct.pack("<hh", *_pack_uv(vertex.get("uv2") or (0.0, 0.0), uv2_log))

    _align_buffer(geom, 2)
    index_data_offset = len(geom)
    geom += struct.pack(f"<{len(indices)}H", *[int(index) & 0xFFFF for index in indices])

    force_skin = bool(original_flags & SUBSET_FLAG_SKINNED) or any(vertex.get("weights") for vertex in vertices)
    if force_skin:
        repaired_count, fallback_count = _repair_missing_skin_weights(vertices, indices)
        if repaired_count:
            detail = (
                f"; {fallback_count} disconnected export vertices used the object's dominant bone"
                if fallback_count else ""
            )
            _append_export_warning(
                export_warnings,
                f"{obj.name}: automatically repaired missing bone weights on {repaired_count} vertices{detail}.",
            )
    anim_vert_count = int(anim_vert_count)
    if not 0 <= anim_vert_count <= len(vertices):
        raise ValueError(
            f"{obj.name}'s saved facial-animation setup no longer matches the mesh. Re-import the original "
            "model with Import Shape Keys enabled, then repeat your edits."
        )
    anim_cluster_count = _align(anim_vert_count, SKIN_CLUSTER_VERTEX_COUNT) // SKIN_CLUSTER_VERTEX_COUNT
    skin_data, cluster_headers = _build_skin_sections(vertices, force_skin, anim_cluster_count)
    vertex_skin_offset = 0
    skin_cluster_offset = 0
    if force_skin:
        _align_buffer(geom, SKIN_CLUSTER_WORD_BYTES)
        vertex_skin_offset = len(geom)
        geom += skin_data
        _align_buffer(geom, SKIN_CLUSTER_WORD_BYTES)
        skin_cluster_offset = len(geom)
        geom += cluster_headers

    reuse_original_stats = False
    if original_record:
        try:
            original_index_count, original_vertex_count = struct.unpack_from("<II", original_record, 0)
        except Exception:
            original_index_count = original_vertex_count = -1
        reuse_original_stats = (
            original_index_count == len(indices) and original_vertex_count == len(vertices)
        )

    surface_area = 0.0
    uv_area = 0.0
    longest_edge = 0.0
    if reuse_original_stats:
        surface_area, uv_area = struct.unpack_from("<ff", original_record, MODEL_SUBSET_SURFACE_AREA_OFFSET)
        longest_edge = float(struct.unpack_from("<H", original_record, MODEL_SUBSET_LONGEST_EDGE_OFFSET)[0]) * mpu
    else:
        for i in range(0, len(indices), 3):
            a = vertices[indices[i]]
            b = vertices[indices[i + 1]]
            c = vertices[indices[i + 2]]
            surface_area += _triangle_area(a["co"], b["co"], c["co"])
            uv_area += _triangle_uv_area(a.get("uv0", (0.0, 0.0)), b.get("uv0", (0.0, 0.0)), c.get("uv0", (0.0, 0.0)))
            longest_edge = max(
                longest_edge,
                _vec_len(_vec_sub(a["co"], b["co"])),
                _vec_len(_vec_sub(b["co"], c["co"])),
                _vec_len(_vec_sub(c["co"], a["co"])),
            )

    record = bytearray(original_record if original_record else b"\x00" * MODEL_SUBSET_RECORD_SIZE)
    if len(record) < MODEL_SUBSET_RECORD_SIZE:
        record += b"\x00" * (MODEL_SUBSET_RECORD_SIZE - len(record))
    flags = (original_flags & ~SUBSET_EXPORT_FLAG_CLEAR_MASK)
    if force_skin:
        flags |= SUBSET_FLAG_SKINNED
    if has_uv1:
        flags |= SUBSET_FLAG_HAS_UV1
    if has_uv2:
        flags |= SUBSET_FLAG_HAS_UV2
    if anim_vert_count:
        flags |= SUBSET_FLAG_HAS_ANIM_VERT
    if needs_origin:
        flags |= SUBSET_FLAG_HAS_ORIGIN_OFFSET

    struct.pack_into("<I", record, MODEL_SUBSET_INDEX_COUNT_OFFSET, len(indices))
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_COUNT_OFFSET, len(vertices))
    struct.pack_into("<I", record, 8, 0)
    struct.pack_into("<I", record, MODEL_SUBSET_INDEX_DATA_OFFSET, index_data_offset)
    struct.pack_into("<I", record, 16, 0x0000FFFF)
    struct.pack_into("<H", record, MODEL_SUBSET_FLAGS_OFFSET, flags)
    struct.pack_into("<H", record, MODEL_SUBSET_UV_LOG_OFFSET, uv_log_scales)
    struct.pack_into("<f", record, MODEL_SUBSET_MPU_OFFSET, mpu)
    struct.pack_into("<H", record, MODEL_SUBSET_MATERIAL_INDEX_OFFSET, int(material_index) & 0xFFFF)
    struct.pack_into("<f", record, MODEL_SUBSET_SURFACE_AREA_OFFSET, float(surface_area))
    struct.pack_into("<f", record, MODEL_SUBSET_UV_AREA_OFFSET, float(uv_area))
    struct.pack_into("<f", record, MODEL_SUBSET_FADE_OUT_DIST_OFFSET, 0.0)
    struct.pack_into("<iii", record, MODEL_SUBSET_OBJ_CENTER_OFFSET, int(origin_packed[0]), int(origin_packed[1]), int(origin_packed[2]))
    struct.pack_into("<I", record, MODEL_SUBSET_OBJ_EXTENTS_OFFSET, extents_word)
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_STD_OFFSET, vertex_std_offset)
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_UV12_OFFSET, vertex_uv12_offset)
    struct.pack_into("<I", record, 72, 0)
    struct.pack_into("<I", record, 76, vertex_skin_offset)
    struct.pack_into("<I", record, 80, skin_cluster_offset)
    struct.pack_into("<I", record, 84, custom_stream_index)
    struct.pack_into("<I", record, MODEL_SUBSET_BASE_OFFSET, 0)
    struct.pack_into("<I", record, 92, len(geom))
    struct.pack_into("<I", record, 96, anim_vert_count)
    struct.pack_into("<H", record, 100, anim_cluster_count)
    struct.pack_into("<H", record, 102, 1)
    struct.pack_into("<H", record, MODEL_SUBSET_LONGEST_EDGE_OFFSET, _clamp_u16(longest_edge / max(mpu, 1e-9)))

    stats = {
        "vertex_count": len(vertices),
        "index_count": len(indices),
        "custom_stream_count": len(vertices),
        "skinned": force_skin,
        "center": center,
        "extents": extents,
        "radius": radius,
        "mpu": mpu,
        "anim_vert_count": anim_vert_count,
        "anim_cluster_count": anim_cluster_count,
        "vertices": vertices,
    }
    return bytes(record), bytes(geom), stats


def _build_subset_geometry_chunks(
    obj,
    arm,
    material_index,
    original_record,
    source_joint_count,
    export_warnings=None,
    fallback_morph_targets=None,
):
    original_flags = 0
    if original_record:
        original_flags = struct.unpack_from("<H", original_record, MODEL_SUBSET_FLAGS_OFFSET)[0]

    vertices, indices, has_uv1, has_uv2, morph_targets, anim_vert_count = _export_mesh_vertices(
        obj,
        arm,
        source_joint_count,
        original_flags=original_flags,
        export_warnings=export_warnings,
        fallback_morph_targets=fallback_morph_targets,
    )
    if not vertices or not indices:
        raise ValueError(
            f"{obj.name} has no faces that can be exported. Add or triangulate faces, then export again."
        )
    if len(vertices) > MODEL_MAX_VERTEX_COUNT or any(index >= MODEL_MAX_VERTEX_COUNT for index in indices):
        raise ValueError(
            f"{obj.name} is too large for one game mesh part ({len(vertices)} export vertices). Split it into "
            f"smaller meshes with fewer than {MODEL_MAX_VERTEX_COUNT} export vertices each, then try again."
        )
    return [(vertices, indices, has_uv1, has_uv2, morph_targets, anim_vert_count)]


def _build_geometry_and_subset_blocks(
    mesh_objects,
    arm,
    material_indices,
    template,
    source_joint_count,
    export_warnings=None,
    source_morph_targets_by_subset=None,
):
    subset_block_hash = BLOCK_HASHES["ModelSubset"]
    original_subset_block = template.payload(subset_block_hash) if subset_block_hash in template.blocks else b""
    original_count = len(original_subset_block) // MODEL_SUBSET_RECORD_SIZE

    subset_records = []
    geom_buffer = bytearray()
    stats = []
    subset_index_map = {}
    morph_targets_by_name = {}
    custom_stream_index = 0
    for index, obj in enumerate(mesh_objects):
        original_record = b""
        try:
            old_index_value = obj.get("engine_subset_index", index)
            old_index = int(index if old_index_value is None else old_index_value)
        except Exception:
            old_index = index
        if 0 <= old_index < original_count:
            start = old_index * MODEL_SUBSET_RECORD_SIZE
            original_record = original_subset_block[start:start + MODEL_SUBSET_RECORD_SIZE]

        chunks = _build_subset_geometry_chunks(
            obj,
            arm,
            material_indices[index],
            original_record,
            source_joint_count,
            export_warnings=export_warnings,
            fallback_morph_targets=(source_morph_targets_by_subset or {}).get(int(old_index), []),
        )
        mapped_indices = []
        for chunk_vertices, chunk_indices, has_uv1, has_uv2, chunk_morph_targets, anim_vert_count in chunks:
            _align_buffer(geom_buffer, DAT1_BLOCK_ALIGN)
            geom_base = len(geom_buffer)
            record, geom, subset_stats = _build_subset_geometry_from_data(
                obj,
                arm,
                material_indices[index],
                original_record,
                custom_stream_index,
                chunk_vertices,
                chunk_indices,
                has_uv1,
                has_uv2,
                anim_vert_count=anim_vert_count,
                export_warnings=export_warnings,
            )
            record = bytearray(record)
            struct.pack_into("<I", record, MODEL_SUBSET_BASE_OFFSET, geom_base)
            geom_buffer += geom
            generated_subset_index = len(subset_records)
            mapped_indices.append(generated_subset_index)
            subset_records.append(bytes(record))
            subset_stats["indices"] = chunk_indices
            subset_stats["source_subset_index"] = int(old_index)
            subset_stats["object_name"] = obj.name
            stats.append(subset_stats)
            for target in chunk_morph_targets:
                name = str(target["name"])
                existing = morph_targets_by_name.get(name)
                if existing is None:
                    existing = {
                        "name": name,
                        "hash": int(target["hash"]) & U32_MASK,
                        "source_index": int(target.get("source_index", -1)),
                        "subsets": [],
                    }
                    morph_targets_by_name[name] = existing
                elif int(existing["hash"]) != (int(target["hash"]) & U32_MASK):
                    raise ValueError(
                        f"Facial shape {name!r} is registered differently on separate meshes. Remove and "
                        "register that shape again with the same name on every mesh, then export again."
                    )
                elif int(existing.get("source_index", -1)) != int(target.get("source_index", -1)):
                    raise ValueError(
                        f"Facial shape {name!r} comes from different source slots on separate meshes. Re-import "
                        "with Import Shape Keys enabled and register matching shapes, then export again."
                    )
                existing["subsets"].append({
                    "subset_index": generated_subset_index,
                    "deltas": target["deltas"],
                })
            custom_stream_index += _align(subset_stats["custom_stream_count"], 2)
        subset_index_map[int(old_index)] = mapped_indices
    return (
        b"".join(subset_records),
        bytes(geom_buffer),
        stats,
        subset_index_map,
        list(morph_targets_by_name.values()),
    )


def _msmr_subset_bounds(vertices):
    coords = [tuple(float(value) for value in vertex["co"]) for vertex in vertices]
    mins = tuple(min(value[axis] for value in coords) for axis in range(3))
    maxs = tuple(max(value[axis] for value in coords) for axis in range(3))
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    extents = tuple((maxs[axis] - mins[axis]) * 0.5 for axis in range(3))
    radius = max((_vec_len(_vec_sub(value, center)) for value in coords), default=0.0)
    return center, extents, radius


def _msmr_color_word(obj, source_vertex_index, fallback=0):
    mesh = obj.data
    source_word = int(fallback) & U32_MASK
    source_attribute = mesh.attributes.get("engine_source_vertex_color")
    if (
        source_attribute is not None
        and source_attribute.domain == 'POINT'
        and source_vertex_index < len(source_attribute.data)
    ):
        source_word = int(source_attribute.data[source_vertex_index].value) & U32_MASK

    color_word = None
    color_attribute = mesh.color_attributes.get("MSMR_Color")
    if (
        color_attribute is not None
        and color_attribute.domain == 'POINT'
        and source_vertex_index < len(color_attribute.data)
    ):
        item = color_attribute.data[source_vertex_index]
        try:
            color = item.color_srgb
        except Exception:
            color = item.color
        channels = [_clamp_u8(float(value) * 255.0) for value in color]
        color_word = sum(int(value) << (channel * 8) for channel, value in enumerate(channels))

    mask_word = None
    mask_attributes = [mesh.attributes.get(f"MSMR_Mask_{name}") for name in "RGBA"]
    if all(
        attribute is not None
        and attribute.domain == 'POINT'
        and source_vertex_index < len(attribute.data)
        for attribute in mask_attributes
    ):
        channels = [
            _clamp_u8(float(attribute.data[source_vertex_index].value) * 255.0)
            for attribute in mask_attributes
        ]
        mask_word = sum(int(value) << (channel * 8) for channel, value in enumerate(channels))

    color_changed = color_word is not None and color_word != source_word
    mask_changed = mask_word is not None and mask_word != source_word
    if color_changed:
        return int(color_word) & U32_MASK
    if mask_changed:
        return int(mask_word) & U32_MASK
    return source_word


def _msmr_vertex_stream_key(vertices, colors):
    digest = hashlib.sha1()
    for vertex, color_word in zip(vertices, colors):
        uv0 = vertex.get("uv0") or (0.0, 0.0)
        uv1 = vertex.get("uv1") or (0.0, 0.0)
        digest.update(struct.pack(
            "<3dIi4dI",
            float(vertex["co"][0]),
            float(vertex["co"][1]),
            float(vertex["co"][2]),
            int(vertex["normal_tangent"]) & U32_MASK,
            int(vertex["position_w"]),
            float(uv0[0]),
            float(uv0[1]),
            float(uv1[0]),
            float(uv1[1]),
            int(color_word) & U32_MASK,
        ))
    return len(vertices), digest.digest()


def _msmr_normalized_group_weights(vertices):
    result = []
    for vertex in vertices:
        normalized = _normalize_skin_weights(vertex.get("weights", []))
        normalized = [(int(joint), int(weight)) for joint, weight in normalized if int(weight) > 0]
        result.append(normalized or [(0, 256)])
    return result


def _msmr_source_subset_skin_matches(
    template,
    vertices,
    force_skin,
    subset_index,
    source_joint_count,
):
    subset_hash = BLOCK_HASHES["ModelSubset"]
    if (
        subset_hash not in template.blocks
        or MSMR_MODEL_SKIN_DATA_HASH not in template.blocks
        or MSMR_MODEL_SKIN_BATCH_HASH not in template.blocks
    ):
        return False
    subset_offset, subset_size = template.blocks[subset_hash]
    if not 0 <= int(subset_index) < int(subset_size) // MSMR_SUBSET_RECORD_SIZE:
        return False
    source_subset = parse_msmr_subset(template.data, subset_offset, subset_index)
    if int(source_subset["vertex_count"]) != len(vertices):
        return False
    if not force_skin:
        return int(source_subset["skin_batch_count"]) == 0
    source_grouped = decode_msmr_skin_weights(
        template.data,
        template.blocks,
        source_subset,
        source_joint_count,
    )
    source_by_vertex = [[] for _index in vertices]
    for joint, weights in source_grouped.items():
        for weight, vertex_indices in weights.items():
            for vertex_index in vertex_indices:
                if not 0 <= int(vertex_index) < len(source_by_vertex):
                    return False
                source_by_vertex[int(vertex_index)].append((int(joint), int(weight)))
    source_by_vertex = [tuple(sorted(values)) for values in source_by_vertex]
    exported_by_vertex = [
        tuple(sorted(values))
        for values in _msmr_normalized_group_weights(vertices)
    ]
    return source_by_vertex == exported_by_vertex


def _msmr_source_skin_matches(
    template,
    subset_vertices,
    subset_force_skin,
    source_joint_count,
):
    subset_hash = BLOCK_HASHES["ModelSubset"]
    if subset_hash not in template.blocks:
        return False
    _subset_offset, subset_size = template.blocks[subset_hash]
    if len(subset_vertices) != int(subset_size) // MSMR_SUBSET_RECORD_SIZE:
        return False
    return all(
        _msmr_source_subset_skin_matches(
            template,
            vertices,
            subset_force_skin[subset_index],
            subset_index,
            source_joint_count,
        )
        for subset_index, vertices in enumerate(subset_vertices)
    )


def _build_msmr_skin_blocks(
    subset_vertices,
    subset_indices,
    subset_force_skin,
    use_joint_remap,
    template=None,
    source_joint_count=0,
    source_subset_indices=None,
):
    skin_data = bytearray()
    skin_batches = bytearray()
    joint_remaps = bytearray()
    subset_batch_ranges = []
    skin_cache = {}
    source_range_cache = {}
    source_subset_indices = list(source_subset_indices or range(len(subset_vertices)))

    source_subset_offset = 0
    source_batch_offset = source_batch_size = 0
    source_skin_offset = source_skin_size = 0
    source_remap_offset = 0
    source_total_batches = 0
    source_total_subsets = 0
    if template is not None and BLOCK_HASHES["ModelSubset"] in template.blocks:
        source_subset_offset, source_subset_size = template.blocks[BLOCK_HASHES["ModelSubset"]]
        source_total_subsets = int(source_subset_size) // MSMR_SUBSET_RECORD_SIZE
        source_batch_offset, source_batch_size = template.blocks.get(MSMR_MODEL_SKIN_BATCH_HASH, (0, 0))
        source_skin_offset, source_skin_size = template.blocks.get(MSMR_MODEL_SKIN_DATA_HASH, (0, 0))
        source_remap_offset, _source_remap_size = template.blocks.get(MSMR_MODEL_SKIN_JOINT_REMAP_HASH, (0, 0))
        source_total_batches = int(source_batch_size) // 16

    for subset_index, vertices in enumerate(subset_vertices):
        first_batch = len(skin_batches) // 16
        if not subset_force_skin[subset_index]:
            subset_batch_ranges.append((first_batch, 0))
            continue
        source_subset_index = (
            int(source_subset_indices[subset_index])
            if subset_index < len(source_subset_indices)
            else subset_index
        )
        source_subset = None
        if (
            template is not None
            and source_batch_offset
            and 0 <= source_subset_index < source_total_subsets
        ):
            source_subset = parse_msmr_subset(
                template.data,
                source_subset_offset,
                source_subset_index,
            )
        if (
            source_subset is not None
            and _msmr_source_subset_skin_matches(
                template,
                vertices,
                True,
                source_subset_index,
                source_joint_count,
            )
        ):
            source_range = (
                int(source_subset["first_skin_batch"]),
                int(source_subset["skin_batch_count"]),
            )
            cached_source_range = source_range_cache.get(source_range)
            if cached_source_range is not None:
                subset_batch_ranges.append(cached_source_range)
                continue
            for source_batch_index in range(source_range[0], sum(source_range)):
                source_record_offset = source_batch_offset + source_batch_index * 16
                record = struct.unpack_from("<IIHHHH", template.data, source_record_offset)
                old_data_offset, old_remap_offset, remap_count, unknown, count, first_vertex = record
                next_data_offset = (
                    struct.unpack_from(
                        "<I",
                        template.data,
                        source_batch_offset + (source_batch_index + 1) * 16,
                    )[0]
                    if source_batch_index + 1 < source_total_batches
                    else source_skin_size
                )
                _align_buffer(skin_data, DAT1_BLOCK_ALIGN)
                data_offset = len(skin_data)
                skin_data.extend(template.data[
                    source_skin_offset + int(old_data_offset):
                    source_skin_offset + int(next_data_offset)
                ])
                remap_offset = 0
                if remap_count:
                    _align_buffer(joint_remaps, DAT1_BLOCK_ALIGN)
                    remap_offset = len(joint_remaps)
                    joint_remaps.extend(template.data[
                        source_remap_offset + int(old_remap_offset):
                        source_remap_offset + int(old_remap_offset) + int(remap_count) * 2
                    ])
                skin_batches.extend(struct.pack(
                    "<IIHHHH",
                    data_offset,
                    remap_offset,
                    remap_count,
                    unknown,
                    count,
                    first_vertex,
                ))
            batch_range = (first_batch, source_range[1])
            source_range_cache[source_range] = batch_range
            subset_batch_ranges.append(batch_range)
            continue
        repaired_count, fallback_count = _repair_missing_skin_weights(
            vertices,
            subset_indices[subset_index],
        )
        if repaired_count:
            log_warning(
                "MSMR subset %d repaired %d missing skin weights (%d fallback)",
                subset_index,
                repaired_count,
                fallback_count,
            )
        normalized = _msmr_normalized_group_weights(vertices)
        digest = hashlib.sha1()
        for weights in normalized:
            digest.update(struct.pack("<B", len(weights)))
            for joint, weight in weights:
                digest.update(struct.pack("<HH", int(joint), int(weight)))
        skin_key = (len(normalized), digest.digest())
        cached_range = skin_cache.get(skin_key)
        if cached_range is not None:
            subset_batch_ranges.append(cached_range)
            continue
        groups = [normalized[start:start + 16] for start in range(0, len(normalized), 16)]
        pending = []
        pending_joints = set()

        def flush_pending():
            nonlocal pending, pending_joints
            if not pending:
                return
            palette = sorted(pending_joints) or [0]
            if use_joint_remap:
                if len(palette) > 256:
                    raise ValueError(f"MSMR subset {subset_index} needs more than 256 joints in one skin batch")
                palette_index = {joint: index for index, joint in enumerate(palette)}
            else:
                if any(not 0 <= int(joint) <= 0xFF for joint in palette):
                    raise ValueError(
                        f"MSMR subset {subset_index} uses a joint above 255, but its source model has no "
                        "joint-remap stream"
                    )
                palette_index = {joint: int(joint) for joint in palette}
            _align_buffer(skin_data, DAT1_BLOCK_ALIGN)
            data_offset = len(skin_data)
            first_vertex = pending[0][0]
            vertex_count = sum(len(group) for _start, group in pending)
            for _group_start, group in pending:
                influence_count = max(len(weights) for weights in group)
                if influence_count > 1:
                    influence_count = max(2, influence_count)
                skin_data.append(influence_count - 1)
                for weights in group:
                    entries = list(weights)
                    if influence_count == 1:
                        skin_data.append(palette_index[int(entries[0][0])] & 0xFF)
                        continue

                    # MSMR stores explicit byte weights for every influence,
                    # while Blender's normalized values total 256. Preserve
                    # that total by making the first byte the residual of all
                    # following bytes. A fully weighted vertex cannot encode
                    # 256 in one byte, so the game convention is 255 + 1 on
                    # the same joint.
                    joints = [int(entries[0][0])] * influence_count
                    encoded_weights = [0] * influence_count
                    encoded_weights[0] = 256
                    for influence_index, (joint, weight) in enumerate(
                        entries[1:influence_count],
                        start=1,
                    ):
                        joints[influence_index] = int(joint)
                        encoded_weights[influence_index] = int(weight)
                        encoded_weights[0] -= int(weight)
                    if encoded_weights[0] > 255:
                        encoded_weights[0] = 255
                        encoded_weights[1] = 1
                        joints[1] = joints[0]
                    elif encoded_weights[0] < 0:
                        encoded_weights[0] = 0
                    for influence_index in range(1, influence_count):
                        if encoded_weights[influence_index] == 0:
                            joints[influence_index] = joints[influence_index - 1]
                    for joint, weight in zip(joints, encoded_weights):
                        skin_data.append(palette_index[int(joint)] & 0xFF)
                        skin_data.append(int(weight) & 0xFF)
            if use_joint_remap:
                _align_buffer(joint_remaps, DAT1_BLOCK_ALIGN)
            remap_offset = len(joint_remaps) if use_joint_remap else 0
            if use_joint_remap:
                joint_remaps.extend(struct.pack(f"<{len(palette)}H", *palette))
            skin_batches.extend(struct.pack(
                "<IIHHHH",
                data_offset,
                remap_offset,
                len(palette) if use_joint_remap else 0,
                0,
                vertex_count,
                first_vertex,
            ))
            pending = []
            pending_joints = set()

        for group_index, group in enumerate(groups):
            group_joints = {joint for weights in group for joint, weight in weights if weight > 0}
            pending_vertex_count = sum(len(pending_group) for _start, pending_group in pending)
            if pending and (
                pending_vertex_count + len(group) > MSMR_SKIN_BATCH_MAX_VERTEX_COUNT
                or (use_joint_remap and len(pending_joints | group_joints) > 256)
            ):
                flush_pending()
            pending.append((group_index * 16, group))
            pending_joints.update(group_joints)
        flush_pending()
        batch_count = len(skin_batches) // 16 - first_batch
        if batch_count > 0xFF or first_batch > 0xFFFF:
            raise ValueError(f"MSMR subset {subset_index} has too many skin batches")
        batch_range = (first_batch, batch_count)
        skin_cache[skin_key] = batch_range
        subset_batch_ranges.append(batch_range)

    return bytes(skin_data), bytes(skin_batches), bytes(joint_remaps), subset_batch_ranges


def _fit_msmr_position_quantization(vertices, source_offset, source_scale):
    coords = [tuple(float(value) for value in vertex["co"]) for vertex in vertices]
    source_offset = tuple(float(value) for value in source_offset)
    source_scale = max(float(source_scale), 1.0e-12)
    if all(
        -32768 <= _round_engine((co[axis] - source_offset[axis]) / source_scale) <= 32767
        for co in coords
        for axis in range(3)
    ):
        return source_offset, source_scale
    mins = tuple(min(co[axis] for co in coords) for axis in range(3))
    maxs = tuple(max(co[axis] for co in coords) for axis in range(3))
    offset = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    scale = max(
        source_scale,
        max((maxs[axis] - mins[axis]) / 65534.0 for axis in range(3)),
    )
    return offset, math.nextafter(scale, math.inf)


def _matrix_is_identity(matrix, threshold=1.0e-7):
    for row in range(4):
        for column in range(4):
            expected = 1.0 if row == column else 0.0
            if abs(float(matrix[row][column]) - expected) > threshold:
                return False
    return True


def _msmr_source_slot_topology(obj, arm, source_geometry, source_subset):
    """Map a compatible mesh back onto its original MSMR vertex slots."""
    mesh = obj.data
    source_vertex_count = int(source_subset["vertex_count"])
    if len(mesh.vertices) > source_vertex_count:
        return None
    if not _matrix_is_identity(_safe_matrix_relative_to_armature(arm, obj)):
        return None

    attributes = {}
    for name in (
        "engine_source_position",
        "engine_source_normal_tangent",
        "engine_position_w",
        "engine_source_uv0_u",
        "engine_source_uv0_v",
    ):
        attribute = mesh.attributes.get(name)
        if attribute is None or attribute.domain != 'POINT' or len(attribute.data) != len(mesh.vertices):
            return None
        attributes[name] = attribute.data

    uv0_layer = _uv_layer_by_name_or_index(mesh, "UV0", 0)
    if uv0_layer is None:
        return None
    source_start = int(source_subset["vertex_start"])
    source_positions = source_geometry["positions"]
    source_normals = source_geometry["normal_words"]
    source_position_ws = source_geometry["position_ws"]
    source_uv0 = source_geometry["uv0"]

    def source_key(local_index):
        global_index = source_start + int(local_index)
        position = source_positions[global_index]
        return struct.pack(
            "<3fIi2f",
            float(position[0]),
            -float(position[2]),
            float(position[1]),
            int(source_normals[global_index]) & U32_MASK,
            int(source_position_ws[global_index]),
            float(source_uv0[global_index][0]),
            float(source_uv0[global_index][1]),
        )

    source_keys = [source_key(local_index) for local_index in range(source_vertex_count)]
    current_keys = []
    for vertex_index in range(len(mesh.vertices)):
        source_position = attributes["engine_source_position"][vertex_index].vector
        current_keys.append(struct.pack(
            "<3fIi2f",
            float(source_position[0]),
            float(source_position[1]),
            float(source_position[2]),
            int(attributes["engine_source_normal_tangent"][vertex_index].value) & U32_MASK,
            int(attributes["engine_position_w"][vertex_index].value),
            float(attributes["engine_source_uv0_u"][vertex_index].value),
            float(attributes["engine_source_uv0_v"][vertex_index].value),
        ))

    slot_map = None
    source_index_attribute = mesh.attributes.get("engine_source_vertex_index")
    if (
        source_index_attribute is not None
        and source_index_attribute.domain == 'POINT'
        and len(source_index_attribute.data) == len(mesh.vertices)
    ):
        candidate_map = [
            int(source_index_attribute.data[vertex_index].value)
            for vertex_index in range(len(mesh.vertices))
        ]
        if (
            len(set(candidate_map)) == len(candidate_map)
            and all(
                0 <= source_slot < source_vertex_count
                and current_keys[vertex_index] == source_keys[source_slot]
                for vertex_index, source_slot in enumerate(candidate_map)
            )
        ):
            slot_map = candidate_map

    if slot_map is None and len(current_keys) == source_vertex_count and current_keys == source_keys:
        # Existing scenes predate engine_source_vertex_index, but an intact
        # source-sized vertex array still has an exact, unambiguous ID order.
        slot_map = list(range(source_vertex_count))

    if slot_map is None:
        # Blender preserves the relative order of surviving vertices when
        # vertices are deleted. Match that subsequence before falling back to
        # unique packed metadata, which cannot identify duplicate source slots.
        ordered_map = []
        source_cursor = 0
        for key in current_keys:
            while source_cursor < source_vertex_count and source_keys[source_cursor] != key:
                source_cursor += 1
            if source_cursor >= source_vertex_count:
                ordered_map = []
                break
            ordered_map.append(source_cursor)
            source_cursor += 1
        if len(ordered_map) == len(current_keys):
            slot_map = ordered_map

    if slot_map is None:
        source_slots = {}
        for local_index, key in enumerate(source_keys):
            source_slots[key] = local_index if key not in source_slots else None
        candidate_map = [source_slots.get(key) for key in current_keys]
        if (
            all(source_slot is not None for source_slot in candidate_map)
            and len(set(candidate_map)) == len(candidate_map)
        ):
            slot_map = [int(source_slot) for source_slot in candidate_map]
    if slot_map is None:
        return None

    mesh.calc_loop_triangles()
    uv0_values = _prefetch_loop_uvs(uv0_layer, len(mesh.loops))
    remapped_indices = []
    for triangle in mesh.loop_triangles:
        for loop_index in triangle.loops:
            loop_index = int(loop_index)
            vertex_index = int(mesh.loops[loop_index].vertex_index)
            source_uv = (
                float(attributes["engine_source_uv0_u"][vertex_index].value),
                float(attributes["engine_source_uv0_v"][vertex_index].value),
            )
            if not _uv_nearly_equal(uv0_values[loop_index], source_uv, threshold=1.0e-7):
                return None
            remapped_indices.append(slot_map[vertex_index])
    if not remapped_indices or len(remapped_indices) > int(source_subset["index_count"]):
        return None
    return {
        "slot_map": slot_map,
        "indices": remapped_indices,
    }


def _msmr_source_slot_vertex_updates(
    vertices,
    color_words,
    topology,
    source_subset,
    source_geometry,
):
    """Map exported vertices onto source slots and identify changed stream values."""
    slot_map = topology["slot_map"]
    source_start = int(source_subset["vertex_start"])
    source_uv1 = source_geometry.get("uv1")
    source_colors = source_geometry.get("color_words")
    export_slot_map = []
    slot_values = {}
    updates = {}

    def same_vertex(left, right):
        return (
            all(abs(float(a) - float(b)) <= 1.0e-7 for a, b in zip(left["co"], right["co"]))
            and int(left["normal_tangent"]) == int(right["normal_tangent"])
            and int(left["position_w"]) == int(right["position_w"])
            and _uv_nearly_equal(
                left.get("uv0", (0.0, 0.0)),
                right.get("uv0", (0.0, 0.0)),
                1.0e-7,
            )
            and _uv_nearly_equal(
                left.get("uv1") or (0.0, 0.0),
                right.get("uv1") or (0.0, 0.0),
                1.0e-7,
            )
        )

    for export_index, vertex in enumerate(vertices):
        source_control_index = int(vertex.get("source_index", -1))
        if not 0 <= source_control_index < len(slot_map):
            return None
        source_slot = int(slot_map[source_control_index])
        if not 0 <= source_slot < int(source_subset["vertex_count"]):
            return None
        export_slot_map.append(source_slot)
        previous = slot_values.get(source_slot)
        if previous is not None and (
            not same_vertex(previous[0], vertex)
            or int(previous[1]) != int(color_words[export_index])
        ):
            # A source slot cannot represent two newly split loop vertices.
            return None
        slot_values[source_slot] = (vertex, int(color_words[export_index]) & U32_MASK)

    source_index_start = int(source_subset["index_start"])
    source_index_count = int(source_subset["index_count"])
    source_indices = source_geometry["indices"][
        source_index_start:source_index_start + source_index_count
    ].astype(np.uint16).tolist()
    if not (int(source_subset["flags"]) & 0x10):
        source_indices = [
            (int(index) - source_start) & 0xFFFF
            for index in source_indices
        ]
    topology_unchanged = source_indices == [int(index) for index in topology["indices"]]
    subset_position_changed = any(
        any(
            abs(float(vertex["co"][axis]) - float(source_geometry["positions"][source_start + source_slot][axis]))
            > 1.0e-7
            for axis in range(3)
        )
        for source_slot, (vertex, _color_word) in slot_values.items()
    )

    for source_slot, (vertex, color_word) in slot_values.items():
        global_index = source_start + source_slot
        co = vertex["co"]
        position_changed = any(
            abs(float(co[axis]) - float(source_geometry["positions"][global_index][axis])) > 1.0e-7
            for axis in range(3)
        )
        tangent_frame_changed = (
            (int(vertex["normal_tangent"]) & U32_MASK)
            != (int(source_geometry["normal_words"][global_index]) & U32_MASK)
            or int(vertex["position_w"]) != int(source_geometry["position_ws"][global_index])
        )
        changed = position_changed or (
            tangent_frame_changed
            and (topology_unchanged or subset_position_changed)
        )
        if source_uv1 is not None:
            changed = changed or not _uv_nearly_equal(
                vertex.get("uv1") or (0.0, 0.0),
                source_uv1[global_index],
                1.0e-7,
            )
        if source_colors is not None:
            changed = changed or color_word != int(source_colors[global_index]) & U32_MASK
        if changed:
            updates[global_index] = (vertex, color_word)

    return {
        "export_slot_map": export_slot_map,
        "vertex_updates": updates,
    }


def _msmr_hybrid_morph_targets(
    template,
    imported_targets,
    imported_source_indices,
    replaced_indices,
    source_slot_topologies=None,
):
    source = decode_msmr_morphs(template.data, template.blocks)
    if source is None:
        return []
    targets = []
    by_source_index = {}
    for source_index, source_target in enumerate(source.get("targets", [])):
        target = {
            "name": str(source_target.get("name", f"Morph_{source_index}")),
            "hash": int(source_target.get("hash", 0)) & U32_MASK,
            "source_index": source_index,
            "subsets": [
                {
                    "subset_index": int(subset.get("subset_index", -1)),
                    "deltas": dict(subset.get("deltas", {})),
                }
                for subset in source_target.get("subsets", [])
                if subset.get("deltas")
            ],
        }
        targets.append(target)
        by_source_index[source_index] = target

    source_slot_topologies = list(source_slot_topologies or [])
    for generated_index in replaced_indices:
        source_subset_index = int(imported_source_indices[generated_index])
        for target in targets:
            target["subsets"] = [
                subset for subset in target["subsets"]
                if int(subset.get("subset_index", -1)) != source_subset_index
            ]
        for edited in imported_targets[generated_index]:
            source_target_index = int(edited.get("source_index", -1))
            target = by_source_index.get(source_target_index)
            if target is None:
                target = {
                    "name": str(edited.get("name", f"Morph_{len(targets)}")),
                    "hash": int(edited.get("hash", 0)) & U32_MASK,
                    "source_index": source_target_index,
                    "subsets": [],
                }
                targets.append(target)
                if source_target_index >= 0:
                    by_source_index[source_target_index] = target
            deltas = dict(edited.get("deltas", {}))
            topology = (
                source_slot_topologies[generated_index]
                if generated_index < len(source_slot_topologies)
                else None
            )
            if topology is not None:
                export_slot_map = topology.get("export_slot_map", [])
                mapped_deltas = {}
                for export_index, delta in deltas.items():
                    export_index = int(export_index)
                    if not 0 <= export_index < len(export_slot_map):
                        raise ValueError(
                            f"MSMR subset {source_subset_index} morph references a missing export vertex"
                        )
                    source_slot = int(export_slot_map[export_index])
                    previous = mapped_deltas.get(source_slot)
                    delta = tuple(float(value) for value in delta)
                    if previous is not None and any(
                        abs(float(left) - float(right)) > 1.0e-7
                        for left, right in zip(previous, delta)
                    ):
                        raise ValueError(
                            f"MSMR subset {source_subset_index} morph splits one source vertex into "
                            "different deltas"
                        )
                    mapped_deltas[source_slot] = delta
                deltas = mapped_deltas
            if deltas:
                target["subsets"].append({
                    "subset_index": source_subset_index,
                    "deltas": deltas,
                })
    return targets


def _msmr_lod0_subset_ids(template):
    look = template.payload(BLOCK_HASHES["ModelLook"])
    if len(look) % 32:
        raise ValueError("The original MSMR Model Look block is truncated")
    return sorted({
        subset_index
        for record_offset in range(0, len(look), 32)
        for start, count in (struct.unpack_from("<HH", look, record_offset),)
        for subset_index in range(int(start), int(start) + int(count))
    })


def _msmr_compact_lod_subset_bases(template, lod0_subset_count):
    """Return a distinct cloned-subset base for every populated MSMR LOD."""
    look = template.payload(BLOCK_HASHES["ModelLook"])
    if len(look) % 32:
        raise ValueError("The original MSMR Model Look block is truncated")
    lod0_subset_count = int(lod0_subset_count)
    bases = [0]
    next_base = lod0_subset_count
    for lod_index in range(1, MSMR_LOOK_LOD_COUNT):
        populated = any(
            struct.unpack_from("<H", look, record_offset + lod_index * 4 + 2)[0]
            for record_offset in range(0, len(look), 32)
        )
        bases.append(next_base if populated else None)
        if populated:
            next_base += lod0_subset_count
    return tuple(bases)


def _build_msmr_geometry_blocks(
    mesh_objects,
    arm,
    material_indices,
    template,
    source_joint_count,
    export_warnings=None,
    source_morph_targets_by_subset=None,
    compact_lod0=False,
):
    original_subset_block = template.payload(BLOCK_HASHES["ModelSubset"])
    original_count = len(original_subset_block) // MSMR_SUBSET_RECORD_SIZE
    subset_records = []
    subset_vertices = []
    subset_indices = []
    subset_color_words = []
    subset_morph_targets = []
    subset_force_skin = []
    subset_index_map = {}
    morph_targets_by_name = {}
    all_vertices = []
    all_indices = []
    color_words = []
    stats = []
    vertex_stream_cache = {}
    index_stream_cache = {}
    source_geometry = decode_msmr_geometry(template.data, template.blocks)
    source_slot_topologies = []

    for generated_index, obj in enumerate(mesh_objects):
        try:
            old_index = int(obj.get("engine_subset_index", generated_index))
        except Exception:
            old_index = generated_index
        original_record = (
            original_subset_block[old_index * MSMR_SUBSET_RECORD_SIZE:(old_index + 1) * MSMR_SUBSET_RECORD_SIZE]
            if 0 <= old_index < original_count
            else b""
        )
        original_flags = struct.unpack_from("<H", original_record, 36)[0] if original_record else 0
        source_subset = (
            parse_msmr_subset(
                template.data,
                template.blocks[BLOCK_HASHES["ModelSubset"]][0],
                old_index,
            )
            if original_record
            else None
        )
        source_slot_topology = (
            _msmr_source_slot_topology(obj, arm, source_geometry, source_subset)
            if source_subset is not None
            else None
        )
        vertices, indices, _has_uv1, _has_uv2, morph_targets, _anim_vert_count = _export_mesh_vertices(
            obj,
            arm,
            source_joint_count,
            original_flags=original_flags,
            export_warnings=export_warnings,
            fallback_morph_targets=(source_morph_targets_by_subset or {}).get(old_index, []),
        )
        if not vertices or not indices:
            raise ValueError(f"{obj.name} has no faces that can be exported")
        if len(vertices) > MODEL_MAX_VERTEX_COUNT or any(int(index) >= MODEL_MAX_VERTEX_COUNT for index in indices):
            raise ValueError(f"{obj.name} exceeds MSMR's 16-bit per-subset vertex limit")

        subset_colors = [
            _msmr_color_word(obj, int(vertex.get("source_index", 0)))
            for vertex in vertices
        ]
        if source_slot_topology is not None:
            slot_updates = _msmr_source_slot_vertex_updates(
                vertices,
                subset_colors,
                source_slot_topology,
                source_subset,
                source_geometry,
            )
            if slot_updates is None:
                source_slot_topology = None
            else:
                source_slot_topology.update(slot_updates)
        source_slot_topologies.append(source_slot_topology)
        vertex_key = _msmr_vertex_stream_key(vertices, subset_colors)
        vertex_start = vertex_stream_cache.get(vertex_key)
        if vertex_start is None:
            vertex_start = len(all_vertices)
            vertex_stream_cache[vertex_key] = vertex_start
            all_vertices.extend(vertices)
            color_words.extend(subset_colors)

        uses_local_indices = bool(original_flags & 0x10) if original_record else True
        stored_indices = [
            int(index) & 0xFFFF
            if uses_local_indices
            else (int(index) + int(vertex_start)) & 0xFFFF
            for index in indices
        ]
        encoded_stored_indices = struct.pack(
            f"<{len(indices)}H",
            *stored_indices,
        )
        index_key = (len(indices), hashlib.sha1(encoded_stored_indices).digest())
        index_start = index_stream_cache.get(index_key)
        if index_start is None:
            index_start = len(all_indices)
            index_stream_cache[index_key] = index_start
            all_indices.extend(stored_indices)
        subset_vertices.append(vertices)
        subset_indices.append(indices)
        subset_color_words.append(subset_colors)
        subset_morph_targets.append(morph_targets)
        force_skin = bool(original_flags & SUBSET_FLAG_SKINNED) or any(vertex.get("weights") for vertex in vertices)
        subset_force_skin.append(force_skin)
        subset_index_map[old_index] = [generated_index]

        record = bytearray(original_record if original_record else b"\x00" * MSMR_SUBSET_RECORD_SIZE)
        if len(record) < MSMR_SUBSET_RECORD_SIZE:
            record.extend(b"\x00" * (MSMR_SUBSET_RECORD_SIZE - len(record)))
        flags = int(original_flags)
        if not original_record:
            flags |= 0x10
        flags = (flags | SUBSET_FLAG_SKINNED) if force_skin else (flags & ~SUBSET_FLAG_SKINNED)
        struct.pack_into("<IIII", record, 20, vertex_start, index_start, len(indices), len(vertices))
        struct.pack_into("<HH", record, 36, flags & 0xFFFF, int(material_indices[generated_index]) & 0xFFFF)
        subset_records.append(record)

        center, extents, radius = _msmr_subset_bounds(vertices)
        stats.append({
            "vertex_count": len(vertices),
            "index_count": len(indices),
            "center": center,
            "extents": extents,
            "radius": radius,
            "vertices": vertices,
            "indices": indices,
            "source_subset_index": old_index,
        })
        for target in morph_targets:
            key = (int(target.get("source_index", -1)), int(target["hash"]) & U32_MASK, str(target["name"]))
            merged = morph_targets_by_name.setdefault(key, {
                "name": str(target["name"]),
                "hash": int(target["hash"]) & U32_MASK,
                "source_index": int(target.get("source_index", -1)),
                "subsets": [],
            })
            merged["subsets"].append({
                "subset_index": generated_index,
                "deltas": target["deltas"],
            })

    imported_source_indices = [
        int(stat.get("source_subset_index", -1))
        for stat in stats
    ]
    if compact_lod0:
        expected_lod0_subset_ids = _msmr_lod0_subset_ids(template)
        if imported_source_indices != expected_lod0_subset_ids:
            raise ValueError(
                "Compact LOD0 export needs every source LOD0 subset exactly once. "
                "Re-import the model without 'Import All LODs', keep every imported mesh parented to its "
                "armature, then export again."
            )
    preserve_source_vertex_layout = (
        not compact_lod0
        and len(imported_source_indices) <= original_count
        and len(set(imported_source_indices)) == len(imported_source_indices)
        and all(0 <= subset_index < original_count for subset_index in imported_source_indices)
        and all(topology is not None for topology in source_slot_topologies)
        and all(not topology.get("vertex_updates") for topology in source_slot_topologies)
        and (
            len(imported_source_indices) < original_count
            or any(
                len(topology["slot_map"])
                < parse_msmr_subset(
                    template.data,
                    template.blocks[BLOCK_HASHES["ModelSubset"]][0],
                    imported_source_indices[generated_index],
                )["vertex_count"]
                or len(topology["indices"])
                < parse_msmr_subset(
                    template.data,
                    template.blocks[BLOCK_HASHES["ModelSubset"]][0],
                    imported_source_indices[generated_index],
                )["index_count"]
                for generated_index, topology in enumerate(source_slot_topologies)
            )
        )
    )
    if preserve_source_vertex_layout:
        preserved_indices = source_geometry["indices"].astype(np.uint16).tolist()
        preserved_records = [
            bytearray(original_subset_block[
                subset_index * MSMR_SUBSET_RECORD_SIZE:(subset_index + 1) * MSMR_SUBSET_RECORD_SIZE
            ])
            for subset_index in range(original_count)
        ]
        for generated_index, topology in enumerate(source_slot_topologies):
            subset_index = imported_source_indices[generated_index]
            source_subset = parse_msmr_subset(
                template.data,
                template.blocks[BLOCK_HASHES["ModelSubset"]][0],
                subset_index,
            )
            stored_indices = [
                int(index) & 0xFFFF
                if int(source_subset["flags"]) & 0x10
                else (int(index) + int(source_subset["vertex_start"])) & 0xFFFF
                for index in topology["indices"]
            ]
            index_start = int(source_subset["index_start"])
            preserved_indices[index_start:index_start + len(stored_indices)] = stored_indices

            record = preserved_records[subset_index]
            struct.pack_into("<I", record, 28, len(stored_indices))
            struct.pack_into("<H", record, 38, int(material_indices[generated_index]) & 0xFFFF)

        built = template.payload(BLOCK_HASHES["ModelBuilt"])
        position_offset = (
            tuple(float(value) for value in struct.unpack_from("<3f", built, 28))
            if len(built) >= 40
            else tuple(source_geometry["position_offset"])
        )
        position_scale = (
            float(struct.unpack_from("<f", built, 44)[0])
            if len(built) >= 48
            else float(source_geometry["position_scale"])
        )
        source_uv_logs = struct.unpack_from("<I", built, 48)[0] if len(built) >= 52 else 0
        replacements = {
            BLOCK_HASHES["ModelSubset"]: b"".join(bytes(record) for record in preserved_records),
            MSMR_MODEL_INDEX_HASH: encode_msmr_index_stream(preserved_indices),
            MSMR_MODEL_STD_VERT_HASH: template.payload(MSMR_MODEL_STD_VERT_HASH),
        }
        return {
            "replacements": replacements,
            "stats": stats,
            "subset_index_map": {
                subset_index: [subset_index]
                for subset_index in range(original_count)
            },
            "morph_targets": [],
            "position_offset": position_offset,
            "position_scale": position_scale,
            "uv_logs": source_uv_logs,
            "vertex_count": len(source_geometry["positions"]),
            "index_count": len(source_geometry["indices"]),
            "source_geometry_unchanged": (
                replacements[MSMR_MODEL_INDEX_HASH] == template.payload(MSMR_MODEL_INDEX_HASH)
            ),
            "source_vertex_layout_preserved": True,
            "source_subset_layout_preserved": True,
            "preserve_source_morphs": True,
        }

    # Keep the complete source subset table (including unimported LODs) when
    # only some imported subsets require new vertex allocations. Compatible
    # subsets continue to reference their original vertex and skin ranges;
    # incompatible subsets are appended to the shared streams and get new
    # skin batches. This avoids rebuilding or renumbering otherwise untouched
    # LOD geometry merely because one visible subset was replaced or joined.
    preserve_source_subset_layout = (
        not compact_lod0
        and len(imported_source_indices) <= original_count
        and len(set(imported_source_indices)) == len(imported_source_indices)
        and all(0 <= subset_index < original_count for subset_index in imported_source_indices)
        and any(
            topology is None or topology.get("vertex_updates")
            for topology in source_slot_topologies
        )
    )
    if preserve_source_subset_layout:
        preserved_records = [
            bytearray(original_subset_block[
                subset_index * MSMR_SUBSET_RECORD_SIZE:(subset_index + 1) * MSMR_SUBSET_RECORD_SIZE
            ])
            for subset_index in range(original_count)
        ]
        preserved_indices = source_geometry["indices"].astype(np.uint16).tolist()
        source_vertex_count = len(source_geometry["positions"])
        appended_vertices = []
        appended_colors = []
        rebuilt_generated_indices = []
        changed_generated_indices = []
        source_vertex_updates = {}

        for generated_index, topology in enumerate(source_slot_topologies):
            subset_index = imported_source_indices[generated_index]
            source_subset = parse_msmr_subset(
                template.data,
                template.blocks[BLOCK_HASHES["ModelSubset"]][0],
                subset_index,
            )
            record = preserved_records[subset_index]
            struct.pack_into(
                "<H",
                record,
                38,
                int(material_indices[generated_index]) & 0xFFFF,
            )
            if topology is not None:
                vertex_updates = dict(topology.get("vertex_updates", {}))
                if vertex_updates:
                    changed_generated_indices.append(generated_index)
                    source_vertex_updates.update(vertex_updates)
                stored_indices = [
                    int(index) & 0xFFFF
                    if int(source_subset["flags"]) & 0x10
                    else (int(index) + int(source_subset["vertex_start"])) & 0xFFFF
                    for index in topology["indices"]
                ]
                index_start = int(source_subset["index_start"])
                preserved_indices[index_start:index_start + len(stored_indices)] = stored_indices
                struct.pack_into("<I", record, 28, len(stored_indices))
                continue

            vertices = subset_vertices[generated_index]
            indices = subset_indices[generated_index]
            vertex_start = source_vertex_count + len(appended_vertices)
            index_start = len(preserved_indices)
            uses_local_indices = bool(int(source_subset["flags"]) & 0x10)
            if not uses_local_indices and any(
                int(index) + vertex_start > 0xFFFF
                for index in indices
            ):
                raise ValueError(
                    f"MSMR subset {subset_index} uses global 16-bit indices and cannot be appended "
                    "after the source vertex stream"
                )
            stored_indices = [
                int(index) & 0xFFFF
                if uses_local_indices
                else int(index) + vertex_start
                for index in indices
            ]
            preserved_indices.extend(stored_indices)
            appended_vertices.extend(vertices)
            appended_colors.extend(subset_color_words[generated_index])
            rebuilt_generated_indices.append(generated_index)

            flags = int(source_subset["flags"])
            flags = (
                flags | SUBSET_FLAG_SKINNED
                if subset_force_skin[generated_index]
                else flags & ~SUBSET_FLAG_SKINNED
            )
            struct.pack_into(
                "<IIII",
                record,
                20,
                vertex_start,
                index_start,
                len(indices),
                len(vertices),
            )
            struct.pack_into("<H", record, 36, flags & 0xFFFF)

        source_skin_data = bytearray(
            template.payload(MSMR_MODEL_SKIN_DATA_HASH)
            if MSMR_MODEL_SKIN_DATA_HASH in template.blocks
            else b""
        )
        source_skin_batches = bytearray(
            template.payload(MSMR_MODEL_SKIN_BATCH_HASH)
            if MSMR_MODEL_SKIN_BATCH_HASH in template.blocks
            else b""
        )
        source_joint_remaps = bytearray(
            template.payload(MSMR_MODEL_SKIN_JOINT_REMAP_HASH)
            if MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks
            else b""
        )
        if len(source_skin_batches) % 16:
            raise ValueError("The original MSMR Model Skin Batch block is truncated")
        if any(subset_force_skin[index] for index in rebuilt_generated_indices) and (
            MSMR_MODEL_SKIN_DATA_HASH not in template.blocks
            or MSMR_MODEL_SKIN_BATCH_HASH not in template.blocks
        ):
            raise ValueError(
                "The original MSMR model has no skin streams, so weighted meshes cannot be added to it"
            )

        rebuilt_skin_data, rebuilt_skin_batches, rebuilt_joint_remaps, rebuilt_batch_ranges = (
            _build_msmr_skin_blocks(
                [subset_vertices[index] for index in rebuilt_generated_indices],
                [subset_indices[index] for index in rebuilt_generated_indices],
                [subset_force_skin[index] for index in rebuilt_generated_indices],
                MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks,
                source_joint_count=source_joint_count,
            )
        )
        skin_data_base = len(source_skin_data)
        if rebuilt_skin_data:
            _align_buffer(source_skin_data, DAT1_BLOCK_ALIGN)
            skin_data_base = len(source_skin_data)
            source_skin_data.extend(rebuilt_skin_data)
        joint_remap_base = len(source_joint_remaps)
        if rebuilt_joint_remaps:
            _align_buffer(source_joint_remaps, DAT1_BLOCK_ALIGN)
            joint_remap_base = len(source_joint_remaps)
            source_joint_remaps.extend(rebuilt_joint_remaps)
        first_new_batch = len(source_skin_batches) // 16
        for batch_offset in range(0, len(rebuilt_skin_batches), 16):
            (
                data_offset,
                remap_offset,
                remap_count,
                unknown,
                count,
                first_vertex,
            ) = struct.unpack_from("<IIHHHH", rebuilt_skin_batches, batch_offset)
            source_skin_batches.extend(struct.pack(
                "<IIHHHH",
                int(data_offset) + skin_data_base,
                int(remap_offset) + joint_remap_base if remap_count else 0,
                remap_count,
                unknown,
                count,
                first_vertex,
            ))
        for generated_index, (relative_first, batch_count) in zip(
            rebuilt_generated_indices,
            rebuilt_batch_ranges,
        ):
            subset_index = imported_source_indices[generated_index]
            first_batch = first_new_batch + int(relative_first) if batch_count else 0
            if first_batch > 0xFFFF or int(batch_count) > 0xFF:
                raise ValueError(f"MSMR subset {subset_index} has too many skin batches")
            struct.pack_into("<H", preserved_records[subset_index], 40, first_batch)
            preserved_records[subset_index][42] = int(batch_count)

        built = template.payload(BLOCK_HASHES["ModelBuilt"])
        source_offset = (
            struct.unpack_from("<3f", built, 28)
            if len(built) >= 40
            else tuple(source_geometry["position_offset"])
        )
        source_scale = (
            struct.unpack_from("<f", built, 44)[0]
            if len(built) >= 48
            else float(source_geometry["position_scale"])
        )
        fit_vertices = []
        for vertex_index in range(source_vertex_count):
            updated = source_vertex_updates.get(vertex_index)
            fit_vertices.append(
                updated[0]
                if updated is not None
                else {
                    "co": tuple(
                        float(value)
                        for value in source_geometry["positions"][vertex_index]
                    )
                }
            )
        fit_vertices.extend(appended_vertices)
        position_offset, position_scale = _fit_msmr_position_quantization(
            fit_vertices,
            source_offset,
            source_scale,
        )
        source_uv_logs = struct.unpack_from("<I", built, 48)[0] if len(built) >= 52 else 0
        uv_log = max(
            source_uv_logs & 0xF,
            _uv_log_for_values(appended_vertices, "uv0"),
        )
        total_vertex_count = source_vertex_count + len(appended_vertices)
        values = np.empty((6, total_vertex_count), dtype=np.int16)
        normal_words = np.empty(total_vertex_count, dtype=np.uint32)
        for vertex_index in range(source_vertex_count):
            updated = source_vertex_updates.get(vertex_index)
            vertex = updated[0] if updated is not None else None
            co = vertex["co"] if vertex is not None else source_geometry["positions"][vertex_index]
            values[0:3, vertex_index] = [
                _clamp_i16((float(co[axis]) - position_offset[axis]) / position_scale)
                for axis in range(3)
            ]
            values[3, vertex_index] = _clamp_i16(
                vertex["position_w"]
                if vertex is not None
                else source_geometry["position_ws"][vertex_index]
            )
            values[4:6, vertex_index] = _pack_uv(
                vertex.get("uv0", (0.0, 0.0))
                if vertex is not None
                else source_geometry["uv0"][vertex_index],
                uv_log,
            )
            normal_words[vertex_index] = (
                int(vertex["normal_tangent"])
                if vertex is not None
                else int(source_geometry["normal_words"][vertex_index])
            ) & U32_MASK
        for appended_index, vertex in enumerate(appended_vertices, start=source_vertex_count):
            co = vertex["co"]
            values[0:3, appended_index] = [
                _clamp_i16((float(co[axis]) - position_offset[axis]) / position_scale)
                for axis in range(3)
            ]
            values[3, appended_index] = _clamp_i16(vertex["position_w"])
            values[4:6, appended_index] = _pack_uv(vertex.get("uv0", (0.0, 0.0)), uv_log)
            normal_words[appended_index] = int(vertex["normal_tangent"]) & U32_MASK

        replacements = {
            BLOCK_HASHES["ModelSubset"]: b"".join(bytes(record) for record in preserved_records),
            MSMR_MODEL_INDEX_HASH: encode_msmr_index_stream(preserved_indices),
            MSMR_MODEL_STD_VERT_HASH: encode_msmr_vertex_stream(normal_words, values),
        }
        if MSMR_MODEL_SKIN_DATA_HASH in template.blocks:
            replacements[MSMR_MODEL_SKIN_DATA_HASH] = bytes(source_skin_data)
        if MSMR_MODEL_SKIN_BATCH_HASH in template.blocks:
            replacements[MSMR_MODEL_SKIN_BATCH_HASH] = bytes(source_skin_batches)
        if MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks:
            replacements[MSMR_MODEL_SKIN_JOINT_REMAP_HASH] = bytes(source_joint_remaps)
        if MSMR_MODEL_UV1_VERT_HASH in template.blocks:
            source_uv1 = source_geometry.get("uv1")
            if source_uv1 is None:
                raise ValueError("The original MSMR UV1 stream could not be decoded")
            uv1_values = np.empty((total_vertex_count, 2), dtype="<i2")
            for vertex_index in range(source_vertex_count):
                updated = source_vertex_updates.get(vertex_index)
                vertex = updated[0] if updated is not None else None
                uv1 = (
                    vertex.get("uv1") or (0.0, 0.0)
                    if vertex is not None
                    else source_uv1[vertex_index]
                )
                uv1_values[vertex_index] = (
                    _clamp_i16(float(uv1[0]) * 32768.0),
                    _clamp_i16(float(uv1[1]) * 32768.0),
                )
            for appended_index, vertex in enumerate(appended_vertices, start=source_vertex_count):
                uv1 = vertex.get("uv1") or (0.0, 0.0)
                uv1_values[appended_index] = (
                    _clamp_i16(float(uv1[0]) * 32768.0),
                    _clamp_i16(float(uv1[1]) * 32768.0),
                )
            replacements[MSMR_MODEL_UV1_VERT_HASH] = uv1_values.tobytes()
        if MSMR_MODEL_COL_VERT_HASH in template.blocks:
            source_colors = source_geometry.get("color_words")
            if source_colors is None:
                raise ValueError("The original MSMR color stream could not be decoded")
            hybrid_colors = np.concatenate((
                np.asarray(source_colors, dtype="<u4"),
                np.asarray(appended_colors, dtype="<u4"),
            ))
            for vertex_index, (_vertex, color_word) in source_vertex_updates.items():
                hybrid_colors[vertex_index] = int(color_word) & U32_MASK
            replacements[MSMR_MODEL_COL_VERT_HASH] = hybrid_colors.astype("<u4", copy=False).tobytes()

        return {
            "replacements": replacements,
            "stats": stats,
            "subset_index_map": {
                subset_index: [subset_index]
                for subset_index in range(original_count)
            },
            "morph_targets": _msmr_hybrid_morph_targets(
                template,
                subset_morph_targets,
                imported_source_indices,
                rebuilt_generated_indices + changed_generated_indices,
                source_slot_topologies,
            ),
            "position_offset": position_offset,
            "position_scale": position_scale,
            "uv_logs": (source_uv_logs & ~0xF) | uv_log,
            "vertex_count": total_vertex_count,
            "index_count": len(preserved_indices),
            "source_geometry_unchanged": (
                replacements[MSMR_MODEL_INDEX_HASH] == template.payload(MSMR_MODEL_INDEX_HASH)
                and replacements[MSMR_MODEL_STD_VERT_HASH] == template.payload(MSMR_MODEL_STD_VERT_HASH)
            ),
            "source_vertex_layout_preserved": not rebuilt_generated_indices,
            "source_subset_layout_preserved": True,
            "preserve_source_morphs": False,
        }

    for subset_index, vertices in enumerate(subset_vertices):
        if not subset_force_skin[subset_index]:
            continue
        repaired_count, fallback_count = _repair_missing_skin_weights(
            vertices,
            subset_indices[subset_index],
        )
        if repaired_count:
            log_warning(
                "MSMR subset %d repaired %d missing skin weights (%d fallback)",
                subset_index,
                repaired_count,
                fallback_count,
            )

    preserve_source_skin = (
        all(
            int(stat.get("source_subset_index", -1)) == subset_index
            for subset_index, stat in enumerate(stats)
        )
        and _msmr_source_skin_matches(
            template,
            subset_vertices,
            subset_force_skin,
            source_joint_count,
        )
    )
    if preserve_source_skin:
        skin_data = template.payload(MSMR_MODEL_SKIN_DATA_HASH)
        skin_batches = template.payload(MSMR_MODEL_SKIN_BATCH_HASH)
        joint_remaps = (
            template.payload(MSMR_MODEL_SKIN_JOINT_REMAP_HASH)
            if MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks
            else b""
        )
        batch_ranges = [
            (
                struct.unpack_from("<H", record, 40)[0],
                int(record[42]),
            )
            for record in subset_records
        ]
    else:
        skin_data, skin_batches, joint_remaps, batch_ranges = _build_msmr_skin_blocks(
            subset_vertices,
            subset_indices,
            subset_force_skin,
            MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks,
            template=template,
            source_joint_count=source_joint_count,
            source_subset_indices=[
                int(stat.get("source_subset_index", subset_index))
                for subset_index, stat in enumerate(stats)
            ],
        )
    for record, (first_batch, batch_count) in zip(subset_records, batch_ranges):
        struct.pack_into("<H", record, 40, first_batch)
        record[42] = batch_count

    built = template.payload(BLOCK_HASHES["ModelBuilt"])
    source_offset = struct.unpack_from("<3f", built, 28) if len(built) >= 40 else (0.0, 0.0, 0.0)
    source_scale = struct.unpack_from("<f", built, 44)[0] if len(built) >= 48 else 1.0 / 4096.0
    position_offset, position_scale = _fit_msmr_position_quantization(
        all_vertices,
        source_offset,
        source_scale,
    )
    source_uv_logs = struct.unpack_from("<I", built, 48)[0] if len(built) >= 52 else 0
    uv_log = max(source_uv_logs & 0xF, _uv_log_for_values(all_vertices, "uv0"))

    live_vertex_count = len(all_vertices)
    live_index_count = len(all_indices)
    output_vertex_count = (
        max(live_vertex_count, len(source_geometry["positions"]))
        if compact_lod0
        else live_vertex_count
    )
    output_index_count = (
        max(live_index_count, len(source_geometry["indices"]))
        if compact_lod0
        else live_index_count
    )
    output_indices = list(all_indices)
    if len(output_indices) < output_index_count:
        output_indices.extend([0] * (output_index_count - len(output_indices)))

    values = np.zeros((6, output_vertex_count), dtype=np.int16)
    normal_words = np.zeros(output_vertex_count, dtype=np.uint32)
    for vertex_index, vertex in enumerate(all_vertices):
        co = vertex["co"]
        values[0:3, vertex_index] = [
            _clamp_i16((float(co[axis]) - position_offset[axis]) / position_scale)
            for axis in range(3)
        ]
        values[3, vertex_index] = _clamp_i16(vertex["position_w"])
        values[4:6, vertex_index] = _pack_uv(vertex.get("uv0", (0.0, 0.0)), uv_log)
        normal_words[vertex_index] = int(vertex["normal_tangent"]) & U32_MASK

    subset_payload = b"".join(bytes(record) for record in subset_records)
    if compact_lod0:
        lod_bases = _msmr_compact_lod_subset_bases(template, len(subset_records))
        subset_payload *= sum(base is not None for base in lod_bases)
        subset_payload = subset_payload.ljust(len(original_subset_block), b"\x00")

    def capacity_padded(payload, block_hash):
        if not compact_lod0:
            return payload
        source_size = int(template.blocks.get(block_hash, (0, 0))[1])
        return payload.ljust(max(len(payload), source_size), b"\x00")

    replacements = {
        BLOCK_HASHES["ModelSubset"]: subset_payload,
        MSMR_MODEL_INDEX_HASH: encode_msmr_index_stream(output_indices),
        MSMR_MODEL_STD_VERT_HASH: encode_msmr_vertex_stream(normal_words, values),
    }
    if skin_data or skin_batches:
        if (
            MSMR_MODEL_SKIN_DATA_HASH not in template.blocks
            or MSMR_MODEL_SKIN_BATCH_HASH not in template.blocks
        ):
            raise ValueError(
                "The original MSMR model has no skin streams, so weighted meshes cannot be added to it"
            )
        replacements[MSMR_MODEL_SKIN_DATA_HASH] = capacity_padded(
            skin_data,
            MSMR_MODEL_SKIN_DATA_HASH,
        )
        replacements[MSMR_MODEL_SKIN_BATCH_HASH] = capacity_padded(
            skin_batches,
            MSMR_MODEL_SKIN_BATCH_HASH,
        )
    else:
        if MSMR_MODEL_SKIN_DATA_HASH in template.blocks:
            replacements[MSMR_MODEL_SKIN_DATA_HASH] = capacity_padded(
                b"",
                MSMR_MODEL_SKIN_DATA_HASH,
            )
        if MSMR_MODEL_SKIN_BATCH_HASH in template.blocks:
            replacements[MSMR_MODEL_SKIN_BATCH_HASH] = capacity_padded(
                b"",
                MSMR_MODEL_SKIN_BATCH_HASH,
            )
    if MSMR_MODEL_SKIN_JOINT_REMAP_HASH in template.blocks:
        replacements[MSMR_MODEL_SKIN_JOINT_REMAP_HASH] = capacity_padded(
            joint_remaps,
            MSMR_MODEL_SKIN_JOINT_REMAP_HASH,
        )
    if MSMR_MODEL_UV1_VERT_HASH in template.blocks:
        uv1_values = np.zeros((output_vertex_count, 2), dtype="<i2")
        for vertex_index, vertex in enumerate(all_vertices):
            uv1 = vertex.get("uv1") or (0.0, 0.0)
            uv1_values[vertex_index] = (
                _clamp_i16(float(uv1[0]) * 32768.0),
                _clamp_i16(float(uv1[1]) * 32768.0),
            )
        replacements[MSMR_MODEL_UV1_VERT_HASH] = uv1_values.tobytes()
    if MSMR_MODEL_COL_VERT_HASH in template.blocks:
        output_colors = np.zeros(output_vertex_count, dtype="<u4")
        output_colors[:len(color_words)] = np.asarray(color_words, dtype="<u4")
        replacements[MSMR_MODEL_COL_VERT_HASH] = output_colors.tobytes()

    source_geometry_unchanged = (
        not compact_lod0
        and replacements[MSMR_MODEL_INDEX_HASH] == template.payload(MSMR_MODEL_INDEX_HASH)
        and replacements[MSMR_MODEL_STD_VERT_HASH] == template.payload(MSMR_MODEL_STD_VERT_HASH)
    )

    return {
        "replacements": replacements,
        "stats": stats,
        "subset_index_map": subset_index_map,
        "morph_targets": list(morph_targets_by_name.values()),
        "position_offset": position_offset,
        "position_scale": position_scale,
        "uv_logs": (source_uv_logs & ~0xF) | uv_log,
        "vertex_count": output_vertex_count,
        "index_count": output_index_count,
        "live_vertex_count": live_vertex_count,
        "live_index_count": live_index_count,
        "compact_lod0": bool(compact_lod0),
        "source_geometry_unchanged": source_geometry_unchanged,
        "source_subset_layout_preserved": False,
    }


def _build_msmr_look_blocks(subset_count, string_pool, template, arm, subset_index_map):
    original_look = template.payload(BLOCK_HASHES["ModelLook"])
    original_built = template.payload(BLOCK_HASHES["ModelLookBuilt"])
    source_look_count = len(original_look) // 32
    source_built_count = min(source_look_count, len(original_built) // MODEL_LOOK_BUILT_SIZE)
    looks = _json_list_from_idprop(arm, "engine_model_looks_json")
    groups = _json_list_from_idprop(arm, "engine_model_look_groups_json")
    if not looks:
        looks = [{
            "index": 0,
            "name": "default",
            "name_hash": string_crc32("default"),
            "subset_ids": list(subset_index_map),
            "lods": [{"start": 0, "count": len(subset_index_map)} for _ in range(8)],
        }]

    def source_header(index):
        if not 0 <= int(index) < source_built_count:
            return None
        offset = int(index) * MODEL_LOOK_BUILT_SIZE
        offsets = list(struct.unpack_from("<7Q", original_built, offset))
        counts = list(struct.unpack_from("<6H", original_built, offset + 56))
        return offsets, counts

    def source_section(index, section):
        header = source_header(index)
        if not header:
            return b"", 0
        offsets, counts = header
        start = int(offsets[section])
        if start < 0 or start >= len(original_built):
            return b"", 0
        if section < 6:
            size = int(counts[section]) * 2
        else:
            candidates = [len(original_built)]
            for other_index in range(source_built_count):
                other = source_header(other_index)
                if not other:
                    continue
                candidates.extend(
                    int(value) for value in other[0]
                    if start < int(value) <= len(original_built)
                )
            size = min(candidates) - start
        if size <= 0 or start + size > len(original_built):
            return b"", 0
        return bytes(original_built[start:start + size]), int(counts[section]) if section < 6 else 0

    look_defs = []
    look_block = bytearray()
    for look_index, look_info in enumerate(looks):
        source_ids = [int(value) for value in look_info.get("subset_ids", [])]
        lods = list(look_info.get("lods", []) or [])
        mapped_lods = []
        union = []
        for lod_index in range(8):
            lod = lods[lod_index] if lod_index < len(lods) else {"start": 0, "count": 0}
            start = max(0, int(lod.get("start", 0)))
            count = max(0, int(lod.get("count", 0)))
            mapped = _mapped_subset_ids(
                source_ids[start:start + count],
                subset_index_map,
                subset_count,
                allow_direct=True,
            )
            mapped = sorted(mapped)
            if mapped and mapped != list(range(mapped[0], mapped[0] + len(mapped))):
                raise ValueError(
                    f"MSMR look {look_info.get('name', look_index)!r} LOD {lod_index} is not a contiguous "
                    "subset range. Reorder the model subsets or adjust the look before exporting."
                )
            mapped_lods.append((mapped[0], len(mapped)) if mapped else (0, 0))
            for subset_index in mapped:
                if subset_index not in union:
                    union.append(subset_index)

        source_index = int(look_info.get("index", look_index))
        source_record_offset = source_index * 32
        record = bytearray(
            original_look[source_record_offset:source_record_offset + 32]
            if 0 <= source_index < source_look_count
            else b"\x00" * 32
        )
        for lod_index, (start, count) in enumerate(mapped_lods):
            struct.pack_into("<HH", record, lod_index * 4, start, count)
        look_block += record

        name = str(look_info.get("name", "") or f"Look {look_index}")
        name_hash = int(look_info.get("name_hash", 0) or string_crc32(name)) & U32_MASK
        look_defs.append({
            "name": name,
            "hash": name_hash,
            "offset": string_pool.add(name),
            "ids": sorted(union),
            "source_index": source_index,
        })

    headers_size = len(look_defs) * MODEL_LOOK_BUILT_SIZE
    headers = bytearray()
    data = bytearray()
    for look_index, look_def in enumerate(look_defs):
        ids = look_def["ids"]
        section_offsets = [headers_size + len(data)]
        section_counts = [len(ids)]
        if ids:
            data += struct.pack(f"<{len(ids)}H", *ids)
        for section_index in range(1, 7):
            section, count = source_section(look_def["source_index"], section_index)
            section_offsets.append(headers_size + len(data))
            if section_index < 6:
                section_counts.append(count)
            data += section
        headers += struct.pack(
            "<7Q6H3I",
            *section_offsets,
            *section_counts,
            look_def["hash"],
            look_def["hash"],
            look_def["offset"],
        )

    look_group = _build_look_group_block(groups, len(look_defs), string_pool)
    return bytes(look_block), bytes(headers + data), look_group


def _build_msmr_compact_lod0_look_block(template, lod0_subset_count):
    """Map each populated MSMR LOD to a distinct clone of its LOD0 subsets."""
    original = template.payload(BLOCK_HASHES["ModelLook"])
    if len(original) % 32:
        raise ValueError("The original MSMR Model Look block is truncated")
    lod_bases = _msmr_compact_lod_subset_bases(template, lod0_subset_count)
    compact = bytearray(original)
    for record_offset in range(0, len(compact), 32):
        lod0_start, lod0_count = struct.unpack_from("<HH", original, record_offset)
        for lod_index in range(1, MSMR_LOOK_LOD_COUNT):
            _start, count = struct.unpack_from(
                "<HH", original, record_offset + lod_index * 4
            )
            if not count:
                continue
            mapped_start = int(lod_bases[lod_index]) + int(lod0_start)
            if mapped_start > 0xFFFF or int(lod0_count) > 0xFFFF:
                raise ValueError("Compact MSMR LOD subset indexes exceed the format limit")
            struct.pack_into(
                "<HH",
                compact,
                record_offset + lod_index * 4,
                mapped_start,
                lod0_count,
            )
    return bytes(compact)


def _build_msmr_compact_lod0_look_built_block(template, look):
    """Build compiled subset masks for the cloned compact LOD ranges."""
    original = template.payload(BLOCK_HASHES["ModelLookBuilt"])
    if len(look) % 32:
        raise ValueError("The original MSMR Model Look block is truncated")
    look_count = len(look) // 32
    headers_size = look_count * MODEL_LOOK_BUILT_SIZE
    if len(original) < headers_size:
        raise ValueError("The original MSMR Model Look Built block is truncated")

    headers = []
    section_offsets = set()
    for look_index in range(look_count):
        header_offset = look_index * MODEL_LOOK_BUILT_SIZE
        offsets = tuple(struct.unpack_from("<7Q", original, header_offset))
        headers.append(offsets)
        section_offsets.update(
            int(offset) for offset in offsets if 0 <= int(offset) <= len(original)
        )

    expected_mask_size = MSMR_LOOK_LOD_COUNT * MSMR_LOOK_BUILT_LOD_MASK_SIZE
    compact = bytearray(original)
    for look_index, offsets in enumerate(headers):
        masks_start = int(offsets[6])
        following_offsets = [
            offset for offset in section_offsets if masks_start < offset <= len(original)
        ]
        masks_end = min(following_offsets, default=len(original))
        if masks_start < headers_size or masks_end - masks_start != expected_mask_size:
            raise ValueError(
                "The original MSMR Model Look Built block has an unsupported LOD mask layout"
            )

        look_offset = look_index * 32
        for lod_index in range(MSMR_LOOK_LOD_COUNT):
            start, count = struct.unpack_from(
                "<HH", look, look_offset + lod_index * 4
            )
            if int(start) + int(count) > MSMR_LOOK_BUILT_LOD_MASK_SIZE * 8:
                raise ValueError("Compact MSMR LOD subset indexes exceed the look-mask limit")
            lod_mask = bytearray(MSMR_LOOK_BUILT_LOD_MASK_SIZE)
            for subset_index in range(int(start), int(start) + int(count)):
                lod_mask[subset_index // 8] |= 1 << (subset_index & 7)
            mask_start = masks_start + lod_index * MSMR_LOOK_BUILT_LOD_MASK_SIZE
            compact[mask_start:mask_start + MSMR_LOOK_BUILT_LOD_MASK_SIZE] = lod_mask

    return bytes(compact)


def _build_msmr_model_built_block(template, geometry):
    original = bytearray(template.payload(BLOCK_HASHES["ModelBuilt"]))
    if len(original) < 120:
        original.extend(b"\x00" * (120 - len(original)))
    source_center = struct.unpack_from("<3f", original, 0)
    source_radius = struct.unpack_from("<f", original, 12)[0]
    source_extents = struct.unpack_from("<3f", original, 16)
    source_bounds = None
    if (
        all(math.isfinite(value) for value in source_center + source_extents)
        and math.isfinite(source_radius)
        and source_radius > 0.0
        and all(value >= 0.0 for value in source_extents)
    ):
        source_bounds = (source_center, source_extents, source_radius)
    containment_tolerance = max(float(geometry["position_scale"]) * 2.0, 1.0e-5)
    source_contains_geometry = bool(source_bounds)
    if source_contains_geometry:
        radius_limit_squared = (float(source_radius) + containment_tolerance) ** 2
        for stat in geometry["stats"]:
            for vertex in stat.get("vertices", ()):
                co = tuple(float(value) for value in vertex["co"])
                if any(
                    abs(co[axis] - float(source_center[axis]))
                    > float(source_extents[axis]) + containment_tolerance
                    for axis in range(3)
                ):
                    source_contains_geometry = False
                    break
                if sum(
                    (co[axis] - float(source_center[axis])) ** 2
                    for axis in range(3)
                ) > radius_limit_squared:
                    source_contains_geometry = False
                    break
            if not source_contains_geometry:
                break
    if source_contains_geometry:
        center, extents, radius = source_bounds
    else:
        center, extents, radius = _combine_model_bounds(
            source_bounds,
            geometry["stats"],
            containment_tolerance=containment_tolerance,
        )
    struct.pack_into("<4f", original, 0, *center, radius)
    struct.pack_into("<3f", original, 16, *extents)
    struct.pack_into("<3f", original, 28, *geometry["position_offset"])
    struct.pack_into("<f", original, 44, float(geometry["position_scale"]))
    struct.pack_into("<I", original, 48, int(geometry["uv_logs"]) & U32_MASK)
    struct.pack_into("<II", original, 100, int(geometry["index_count"]), int(geometry["vertex_count"]))
    return bytes(original)


def _build_msmr_morph_blocks(
    template,
    edited_targets,
    string_pool,
):
    source = decode_msmr_morphs(template.data, template.blocks)
    if source is None:
        return None
    edited_by_index = {
        int(target.get("source_index", -1)): target
        for target in edited_targets
        if int(target.get("source_index", -1)) >= 0
    }
    encoded_targets = []
    used_edited = set()
    for source_index, source_target in enumerate(source.get("targets", [])):
        edited = edited_by_index.get(source_index)
        if edited is not None:
            used_edited.add(id(edited))
        target = {
            "name": str((edited or source_target).get("name", source_target.get("name", f"Morph_{source_index}"))),
            "hash": int((edited or source_target).get("hash", source_target.get("hash", 0))) & U32_MASK,
            "subsets": list((edited or {}).get("subsets", [])),
            "packing_kind": int(source_target.get("packing_kind", 1)),
            "packing_null": int(source_target.get("packing_null", 0)),
            "component_bits": int(source_target.get("component_bits", 16)),
            "normal_scale": float(source_target.get("normal_scale", 0.0)),
            "normal_bias": float(source_target.get("normal_bias", 0.0)),
        }
        target["name_offset"] = string_pool.add(target["name"])
        encoded_targets.append(target)
    for edited in edited_targets:
        if id(edited) in used_edited:
            continue
        target = dict(edited)
        target["name_offset"] = string_pool.add(target["name"])
        target.setdefault("component_bits", 16)
        encoded_targets.append(target)
    preserve_source_morphs = len(encoded_targets) == len(source.get("targets", []))
    if preserve_source_morphs:
        for source_target, encoded_target in zip(source.get("targets", []), encoded_targets):
            if int(source_target.get("hash", 0)) != int(encoded_target.get("hash", 0)):
                preserve_source_morphs = False
                break
            source_subsets = {
                int(subset.get("subset_index", -1)): {
                    int(vertex_index): tuple(float(value) for value in delta)
                    for vertex_index, delta in subset.get("deltas", {}).items()
                }
                for subset in source_target.get("subsets", [])
                if subset.get("deltas")
            }
            encoded_subsets = {
                int(subset.get("subset_index", -1)): {
                    int(vertex_index): tuple(float(value) for value in delta)
                    for vertex_index, delta in subset.get("deltas", {}).items()
                }
                for subset in encoded_target.get("subsets", [])
                if subset.get("deltas")
            }
            if source_subsets.keys() != encoded_subsets.keys():
                preserve_source_morphs = False
                break
            for subset_index, source_deltas in source_subsets.items():
                encoded_deltas = encoded_subsets[subset_index]
                if source_deltas.keys() != encoded_deltas.keys():
                    preserve_source_morphs = False
                    break
                if any(
                    any(
                        abs(left - right) > 1.0e-7
                        for left, right in zip(delta, encoded_deltas[vertex_index])
                    )
                    for vertex_index, delta in source_deltas.items()
                ):
                    preserve_source_morphs = False
                    break
            if not preserve_source_morphs:
                break
    if preserve_source_morphs:
        return {
            MSMR_MODEL_ANIM_MORPH_INFO_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_INFO_HASH),
            MSMR_MODEL_ANIM_MORPH_DATA_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_DATA_HASH),
            MSMR_MODEL_ANIM_MORPH_INDICES_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_INDICES_HASH),
        }, len(encoded_targets)
    info, deltas, indices = encode_msmr_morphs(
        encoded_targets,
        source.get("mirrors", []),
        source.get("version", 2),
    )
    return {
        MSMR_MODEL_ANIM_MORPH_INFO_HASH: info,
        MSMR_MODEL_ANIM_MORPH_DATA_HASH: deltas,
        MSMR_MODEL_ANIM_MORPH_INDICES_HASH: indices,
    }, len(encoded_targets)


def _json_list_from_idprop(owner, key):
    try:
        data = json.loads(str(owner.get(key, "[]") or "[]"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _msmr_lod0_scene_looks_match_source(arm, template):
    """Detect a stale modified flag when a LOD0-only scene still matches its source looks."""
    looks = _json_list_from_idprop(arm, "engine_model_looks_json")
    groups = _json_list_from_idprop(arm, "engine_model_look_groups_json")
    source_look = template.payload(BLOCK_HASHES["ModelLook"])
    source_built = template.payload(BLOCK_HASHES["ModelLookBuilt"])
    source_look_count = len(source_look) // 32
    if len(source_look) % 32 or len(looks) != source_look_count:
        return False

    seen_source_indices = set()
    for fallback_index, look in enumerate(looks):
        source_index = int(look.get("index", fallback_index))
        if not 0 <= source_index < source_look_count or source_index in seen_source_indices:
            return False
        seen_source_indices.add(source_index)
        source_start, source_count = struct.unpack_from("<HH", source_look, source_index * 32)
        source_ids = [int(value) for value in look.get("subset_ids", [])]
        lods = list(look.get("lods", []) or [])
        lod0 = lods[0] if lods else {"start": 0, "count": 0}
        start = max(0, int(lod0.get("start", 0)))
        count = max(0, int(lod0.get("count", 0)))
        if source_ids[start:start + count] != list(range(source_start, source_start + source_count)):
            return False
        built_offset = source_index * MODEL_LOOK_BUILT_SIZE
        if built_offset + MODEL_LOOK_BUILT_SIZE <= len(source_built):
            source_name_hash = struct.unpack_from("<I", source_built, built_offset + 72)[0]
            if int(look.get("name_hash", 0)) & U32_MASK != int(source_name_hash) & U32_MASK:
                return False

    try:
        source_groups = parse_msmr_look_groups_metadata(template.data, template.blocks)
    except Exception:
        return False
    if len(groups) != len(source_groups):
        return False
    source_groups_by_index = {
        int(group.get("index", index)): group
        for index, group in enumerate(source_groups)
    }
    for fallback_index, group in enumerate(groups):
        source_index = int(group.get("index", fallback_index))
        source_group = source_groups_by_index.get(source_index)
        if source_group is None:
            return False
        if int(group.get("name_hash", 0)) & U32_MASK != int(source_group.get("name_hash", 0)) & U32_MASK:
            return False
        if [int(value) for value in group.get("look_indices", [])] != [
            int(value) for value in source_group.get("look_indices", [])
        ]:
            return False
    return True


def _mapped_subset_ids(source_ids, subset_index_map, subset_count, allow_direct=False):
    result = []
    for value in source_ids:
        try:
            source_id = int(value)
        except Exception:
            continue
        mapped = subset_index_map.get(source_id)
        if mapped is None and allow_direct and 0 <= source_id < subset_count:
            mapped = source_id
        if mapped is None:
            continue
        mapped_values = mapped if isinstance(mapped, (list, tuple, set)) else [mapped]
        for mapped_value in mapped_values:
            try:
                mapped_index = int(mapped_value)
            except Exception:
                continue
            if 0 <= mapped_index < subset_count and mapped_index not in result:
                result.append(mapped_index)
    return result


def _build_look_group_block(groups, look_count, string_pool):
    if not groups:
        groups = [{
            "name": "default",
            "name_hash": string_crc32("default"),
            "look_indices": list(range(look_count)),
        }]

    records = bytearray()
    indices = bytearray()
    records_base = 1
    indices_base = len(groups) * MODEL_LOOK_GROUP_SIZE
    for group_index, group in enumerate(groups):
        raw_indices = group.get("look_indices", [])
        look_indices = []
        for value in raw_indices:
            try:
                look_index = int(value)
            except Exception:
                continue
            if 0 <= look_index < look_count and look_index not in look_indices:
                look_indices.append(look_index)

        name = str(group.get("name", "") or f"Look Group {group_index}")
        name_hash = int(group.get("name_hash", 0) or 0) & U32_MASK
        if not name_hash:
            name_hash = string_crc32(name)
        name_offset = string_pool.add(name)
        indices_offset = indices_base + len(indices)
        records += struct.pack("<QH6sII", indices_offset, len(look_indices), b"\x00" * 6, name_hash, name_offset)
        if look_indices:
            indices += struct.pack(f"<{len(look_indices)}H", *look_indices)

    return bytes(struct.pack("<B", len(groups)) + records + indices)


def _build_look_blocks(
    subset_count,
    string_pool,
    template,
    arm=None,
    subset_index_map=None,
):
    look_name = "default"
    look_name_offset = string_pool.add(look_name)
    look_hash = string_crc32(look_name)

    original_look = template.payload(BLOCK_HASHES["ModelLook"])
    original_look_built = template.payload(BLOCK_HASHES["ModelLookBuilt"])
    original_subset_block = template.payload(BLOCK_HASHES["ModelSubset"]) if BLOCK_HASHES["ModelSubset"] in template.blocks else b""
    source_subset_count = len(original_subset_block) // MODEL_SUBSET_RECORD_SIZE
    source_look_count = max(1, len(original_look) // MODEL_LOOK_SIZE)
    source_built_look_count = min(source_look_count, len(original_look_built) // MODEL_LOOK_BUILT_SIZE)
    subset_index_map = subset_index_map or {}
    use_custom_looks = bool(arm and arm.get("engine_model_looks_modified", False))
    custom_looks = _json_list_from_idprop(arm, "engine_model_looks_json") if use_custom_looks else []
    custom_groups = _json_list_from_idprop(arm, "engine_model_look_groups_json") if use_custom_looks else []

    def source_look_index(value, fallback=0):
        try:
            index = int(value)
        except Exception:
            index = fallback
        if 0 <= index < source_built_look_count:
            return index
        return 0 if source_built_look_count else -1

    def source_look_header(index):
        index = source_look_index(index)
        if index < 0:
            return None
        offset = index * MODEL_LOOK_BUILT_SIZE
        if offset + MODEL_LOOK_BUILT_SIZE > len(original_look_built):
            return None
        offsets = list(struct.unpack_from("<7Q", original_look_built, offset))
        counts = list(struct.unpack_from("<6H", original_look_built, offset + 56))
        hashes = struct.unpack_from("<3I", original_look_built, offset + 68)
        return offsets, counts, hashes

    def source_look_section_size(index, section):
        header = source_look_header(index)
        if not header:
            return 0
        offsets, counts, _hashes = header
        data_offset = int(offsets[section])
        if data_offset < 0 or data_offset >= len(original_look_built):
            return 0
        if section < 6:
            return max(0, int(counts[section]) * 2)
        candidates = [len(original_look_built)]
        for source_index in range(source_built_look_count):
            other = source_look_header(source_index)
            if not other:
                continue
            for other_offset in other[0]:
                other_offset = int(other_offset)
                if data_offset < other_offset <= len(original_look_built):
                    candidates.append(other_offset)
        return max(0, min(candidates) - data_offset)

    def source_look_section(index, section):
        header = source_look_header(index)
        if not header:
            return b"", 0
        offsets, counts, _hashes = header
        data_offset = int(offsets[section])
        size = source_look_section_size(index, section)
        if size <= 0 or data_offset < 0 or data_offset + size > len(original_look_built):
            return b"", 0
        count = int(counts[section]) if section < 6 else 0
        return bytes(original_look_built[data_offset:data_offset + size]), count

    look_defs = []
    if custom_looks:
        for look_index, look_info in enumerate(custom_looks):
            mapped_ids = _mapped_subset_ids(look_info.get("subset_ids", []), subset_index_map, subset_count, allow_direct=True)
            name = str(look_info.get("name", "") or f"Look {look_index}")
            name_hash = int(look_info.get("name_hash", 0) or 0) & U32_MASK
            if not name_hash:
                name_hash = string_crc32(name)
            look_defs.append({
                "ids": mapped_ids,
                "name": name,
                "hash": name_hash,
                "offset": string_pool.add(name),
                "source_index": source_look_index(look_info.get("index", look_index), look_index),
            })
    if not look_defs:
        for look_index in range(source_look_count):
            original_offset = look_index * MODEL_LOOK_BUILT_SIZE
            mapped_ids = []
            if original_offset + MODEL_LOOK_BUILT_SIZE <= len(original_look_built):
                subset_ids_offset = struct.unpack_from("<Q", original_look_built, original_offset)[0]
                subset_id_count = struct.unpack_from("<H", original_look_built, original_offset + 56)[0]
                original_name_hash, _original_name_hash_lower, original_name_offset = struct.unpack_from(
                    "<3I", original_look_built, original_offset + 68
                )
                ids_start = int(subset_ids_offset)
                ids_end = ids_start + int(subset_id_count) * 2
                if 0 <= ids_start <= ids_end <= len(original_look_built) and subset_id_count:
                    source_ids = list(struct.unpack_from(f"<{int(subset_id_count)}H", original_look_built, ids_start))
                    mapped_ids = _mapped_subset_ids(source_ids, subset_index_map, subset_count)
            else:
                original_name_hash = look_hash
                original_name_offset = look_name_offset
            if not mapped_ids and look_index == 0:
                mapped_ids = [index for index in range(min(source_subset_count, subset_count))]
            look_defs.append({
                "ids": mapped_ids,
                "name": f"Look {look_index}",
                "hash": original_name_hash,
                "offset": original_name_offset,
                "source_index": source_look_index(look_index, look_index),
            })
    look_count = max(1, len(look_defs))

    look = bytearray()
    for look_index, look_def in enumerate(look_defs):
        ids = look_def["ids"]
        source_index = int(look_def.get("source_index", look_index))
        source_offset = source_index * MODEL_LOOK_SIZE
        if 0 <= source_offset and source_offset + MODEL_LOOK_SIZE <= len(original_look):
            record = bytearray(original_look[source_offset:source_offset + MODEL_LOOK_SIZE])
        else:
            record = bytearray(b"\x00" * MODEL_LOOK_SIZE)
        for lod_index in range(8):
            struct.pack_into("<HH", record, lod_index * 4, 0, len(ids))
        look += record

    look_built_headers = bytearray()
    headers_size = look_count * MODEL_LOOK_BUILT_SIZE
    look_built_data = bytearray()
    for look_index, look_def in enumerate(look_defs):
        ids = look_def["ids"]
        source_index = int(look_def.get("source_index", look_index))
        section_offsets = []
        section_counts = []

        section_offsets.append(headers_size + len(look_built_data))
        section_counts.append(len(ids))
        if ids:
            look_built_data += struct.pack(f"<{len(ids)}H", *ids)

        for section in range(1, 7):
            chunk, count = source_look_section(source_index, section)
            section_offsets.append(headers_size + len(look_built_data))
            if section < 6:
                section_counts.append(count)
            if chunk:
                look_built_data += chunk

        name_hash = int(look_def.get("hash", look_hash)) & U32_MASK
        name_offset = int(look_def.get("offset", look_name_offset))
        look_built_headers += struct.pack(
            "<7Q6H3I",
            section_offsets[0],
            section_offsets[1],
            section_offsets[2],
            section_offsets[3],
            section_offsets[4],
            section_offsets[5],
            section_offsets[6],
            section_counts[0],
            section_counts[1],
            section_counts[2],
            section_counts[3],
            section_counts[4],
            section_counts[5],
            name_hash,
            name_hash,
            name_offset,
        )

    if custom_looks:
        look_group = _build_look_group_block(custom_groups, look_count, string_pool)
    else:
        look_group = template.payload(BLOCK_HASHES["ModelLookGroup"])
    if not look_group:
        look_group = bytearray()
        look_group += struct.pack("<B", 1)
        look_group += struct.pack("<QH6sII", MODEL_LOOK_GROUP_SIZE, look_count, b"\x00" * 6, look_hash, look_name_offset)
        look_group += struct.pack(f"<{look_count}H", *range(look_count))
    return bytes(look), bytes(look_built_headers + look_built_data), bytes(look_group)


def _valid_model_bounds(bsphere, aabb):
    return (
        bsphere is not None
        and aabb is not None
        and len(bsphere) == 4
        and len(aabb) == 3
        and all(math.isfinite(float(v)) for v in tuple(bsphere) + tuple(aabb))
        and float(bsphere[3]) > 0.0
        and all(float(v) >= 0.0 for v in aabb)
    )


def _float_list_from_json_prop(owner, key, count):
    if owner is None:
        return None
    raw = owner.get(key)
    if raw is None:
        return None
    try:
        values = json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return None
    if len(values) != count:
        return None
    try:
        values = tuple(float(v) for v in values)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _positive_float_prop(owner, key):
    if owner is None:
        return None
    try:
        value = float(owner.get(key))
    except Exception:
        return None
    if math.isfinite(value) and value > 0.0:
        return value
    return None


def _source_model_built_state(model_built, arm=None):
    bsphere = _float_list_from_json_prop(arm, "engine_model_source_bsphere_json", 4)
    aabb = _float_list_from_json_prop(arm, "engine_model_source_aabb_json", 3)
    source_bounds = (bsphere[:3], aabb, bsphere[3]) if _valid_model_bounds(bsphere, aabb) else None
    source_common_mpu = _positive_float_prop(arm, "engine_model_source_common_mpu")
    source_vertex_mpu = _positive_float_prop(arm, "engine_model_source_vertex_mpu")

    try:
        if source_bounds is None:
            block_bsphere = struct.unpack_from("<4f", model_built, MODEL_BUILT_BSPHERE_OFFSET)
            block_aabb = struct.unpack_from("<3f", model_built, MODEL_BUILT_AABB_EXTENTS_OFFSET)
            if _valid_model_bounds(block_bsphere, block_aabb):
                source_bounds = (block_bsphere[:3], block_aabb, block_bsphere[3])
        if source_common_mpu is None:
            value = struct.unpack_from("<f", model_built, MODEL_BUILT_COMMON_MPU_OFFSET)[0]
            if math.isfinite(value) and value > 0.0:
                source_common_mpu = float(value)
        if source_vertex_mpu is None:
            value = struct.unpack_from("<f", model_built, MODEL_BUILT_VERTEX_MPU_OFFSET)[0]
            if math.isfinite(value) and value > 0.0:
                source_vertex_mpu = float(value)
    except Exception:
        pass

    return source_bounds, source_common_mpu, source_vertex_mpu


def _higher_power_of_two(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _model_mpu_from_bounds(center, extents):
    mins = [float(center[axis]) - float(extents[axis]) for axis in range(3)]
    maxs = [float(center[axis]) + float(extents[axis]) for axis in range(3)]
    min_component = min(mins)
    max_component = max(maxs)

    unbounded_int_min = int(math.ceil(abs(min_component) * float(1 << 15)))
    unbounded_int_max = int(math.ceil(abs(max_component) * float(1 << 15)))
    unbounded_int_range = max(unbounded_int_min, unbounded_int_max + 1)
    max_component_aligned = float(_higher_power_of_two(unbounded_int_range) >> 15)

    vertex_range = max(max_component_aligned, 8.0)
    return vertex_range / float(1 << 15)


def _combine_model_bounds(source_bounds, subset_stats, containment_tolerance=0.0):
    containment_tolerance = max(0.0, float(containment_tolerance))
    bounds = []
    radius_sources = []
    if source_bounds:
        source_center, source_extents, source_radius = source_bounds
        source_center = tuple(float(v) for v in source_center)
        source_extents = tuple(float(v) for v in source_extents)
        source_radius = float(source_radius)
        source_mins = tuple(source_center[axis] - source_extents[axis] for axis in range(3))
        source_maxs = tuple(source_center[axis] + source_extents[axis] for axis in range(3))
        if subset_stats:
            all_inside_source = True
            for stat in subset_stats:
                center = tuple(float(stat["center"][axis]) for axis in range(3))
                extents = tuple(float(stat["extents"][axis]) for axis in range(3))
                radius = float(stat.get("radius", 0.0))
                if any(
                    center[axis] - extents[axis] < source_mins[axis] - containment_tolerance
                    for axis in range(3)
                ):
                    all_inside_source = False
                    break
                if any(
                    center[axis] + extents[axis] > source_maxs[axis] + containment_tolerance
                    for axis in range(3)
                ):
                    all_inside_source = False
                    break
                if (
                    _vec_len(_vec_sub(center, source_center)) + radius
                    > source_radius + containment_tolerance
                ):
                    all_inside_source = False
                    break
            if all_inside_source:
                return source_center, source_extents, source_radius
        bounds.append((
            source_mins,
            source_maxs,
        ))
        radius_sources.append((source_center, source_radius))
    for stat in subset_stats:
        center = tuple(float(stat["center"][axis]) for axis in range(3))
        extents = tuple(float(stat["extents"][axis]) for axis in range(3))
        radius = float(stat.get("radius", 0.0))
        bounds.append((
            tuple(center[axis] - extents[axis] for axis in range(3)),
            tuple(center[axis] + extents[axis] for axis in range(3)),
        ))
        radius_sources.append((center, radius))

    if not bounds:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0

    mins = [min(item[0][axis] for item in bounds) for axis in range(3)]
    maxs = [max(item[1][axis] for item in bounds) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    extents = tuple((maxs[axis] - mins[axis]) * 0.5 for axis in range(3))
    radius_padding = max(MODEL_GLOBAL_BOUNDS_PADDING, max((radius for _center, radius in radius_sources), default=0.0) * 0.01)
    radius = max((_vec_len(_vec_sub(item_center, center)) + item_radius for item_center, item_radius in radius_sources), default=0.0)
    return center, tuple(value + radius_padding for value in extents), radius + radius_padding


def _build_model_built_block(template, subset_stats, arm=None, has_morph=False, has_smooth=False):
    block_hash = BLOCK_HASHES["ModelBuilt"]
    original = bytearray(template.payload(block_hash) if block_hash in template.blocks else b"\x00" * MODEL_BUILT_SIZE)
    if len(original) < MODEL_BUILT_SIZE:
        original += b"\x00" * (MODEL_BUILT_SIZE - len(original))
    model_built = original[:MODEL_BUILT_SIZE]

    flags = struct.unpack_from("<Q", model_built, MODEL_BUILT_FLAGS_OFFSET)[0]
    flags &= ~(MODEL_FLAG_ANIM_VERT | MODEL_FLAG_ANIM_DYNAMICS | MODEL_FLAG_USES_AUTO_LODS)
    if has_morph:
        flags |= MODEL_FLAG_ANIM_VERT
    if not any(stat.get("skinned") for stat in subset_stats):
        flags &= ~(MODEL_FLAG_HAS_SKINNING | MODEL_FLAG_HAS_GPU_SKINNING)
    struct.pack_into("<Q", model_built, MODEL_BUILT_FLAGS_OFFSET, flags)

    content_flags = struct.unpack_from("<H", model_built, MODEL_BUILT_CONTENT_FLAGS_OFFSET)[0]
    content_flags &= ~(
        CONTENT_FLAG_ANIM_MORPH
        | CONTENT_FLAG_ANIM_ZIVA
        | CONTENT_FLAG_ANIM_VERT_SMOOTH
        | CONTENT_FLAG_USES_AUTO_LODS
    )
    if has_morph:
        content_flags |= CONTENT_FLAG_ANIM_MORPH
    if has_smooth:
        content_flags |= CONTENT_FLAG_ANIM_VERT_SMOOTH
    struct.pack_into("<H", model_built, MODEL_BUILT_CONTENT_FLAGS_OFFSET, content_flags)

    source_bounds, source_common_mpu, source_vertex_mpu = _source_model_built_state(model_built, arm=arm)

    center, extents, radius = _combine_model_bounds(source_bounds, subset_stats)

    common_mpu = source_common_mpu
    bounds_mpu = _model_mpu_from_bounds(center, extents) if radius > 0.0 else MODEL_DEFAULT_MPU
    if common_mpu is None:
        common_mpu = bounds_mpu
    else:
        common_mpu = max(common_mpu, bounds_mpu)
    if not (math.isfinite(common_mpu) and common_mpu > 0.0):
        common_mpu = MODEL_DEFAULT_MPU

    subset_vertex_mpus = []
    for stat in subset_stats:
        try:
            stat_mpu = float(stat.get("mpu", 0.0) or 0.0)
        except Exception:
            continue
        if math.isfinite(stat_mpu) and stat_mpu > 0.0:
            subset_vertex_mpus.append(stat_mpu)
    subset_vertex_mpu = max(subset_vertex_mpus) if subset_vertex_mpus else None
    vertex_mpu = subset_vertex_mpu if subset_vertex_mpu is not None else source_vertex_mpu
    if vertex_mpu is None:
        vertex_mpu = common_mpu
    if not (math.isfinite(vertex_mpu) and vertex_mpu > 0.0):
        vertex_mpu = common_mpu

    struct.pack_into("<H", model_built, MODEL_BUILT_FADE_OUT_DIST_OFFSET, 0)
    struct.pack_into("<4f", model_built, MODEL_BUILT_BSPHERE_OFFSET, center[0], center[1], center[2], radius)
    struct.pack_into("<3f", model_built, MODEL_BUILT_AABB_EXTENTS_OFFSET, extents[0], extents[1], extents[2])
    struct.pack_into("<f", model_built, MODEL_BUILT_COMMON_MPU_OFFSET, common_mpu)
    struct.pack_into("<f", model_built, MODEL_BUILT_VERTEX_MPU_OFFSET, vertex_mpu)
    struct.pack_into("<I", model_built, MODEL_BUILT_CUSTOM_STREAM_COUNT_OFFSET, sum(stat["custom_stream_count"] for stat in subset_stats))
    struct.pack_into("<H", model_built, MODEL_BUILT_SUBSET_LOD_MASK_COUNT_OFFSET, len(subset_stats))
    struct.pack_into("<b", model_built, MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET, 0)
    return bytes(model_built)


def _source_morph_targets_for_older_scene(template, mesh_objects):
    decoded = decode_model_morph2(template.data, template.blocks)
    if not decoded:
        return {}, 0
    objects_by_subset = {}
    for obj in mesh_objects:
        try:
            subset_index = int(obj.get("engine_subset_index", -1))
        except Exception:
            continue
        if subset_index >= 0:
            objects_by_subset.setdefault(subset_index, []).append(obj)

    source_subset = template.payload(BLOCK_HASHES["ModelSubset"])
    geom_base, _geom_size = template.blocks[BLOCK_HASHES["ModelSubsetGeomData"]]
    source_subset_count = len(source_subset) // MODEL_SUBSET_RECORD_SIZE
    result = {}
    skipped_records = 0
    for target in decoded.get("targets", []):
        for target_subset in target.get("subsets", []):
            subset_index = int(target_subset["subset_index"])
            objects = objects_by_subset.get(subset_index, [])
            if not objects:
                skipped_records += 1
                continue
            if len(objects) != 1 or not 0 <= subset_index < source_subset_count:
                raise ValueError(
                    f"The saved facial shape {target['name']!r} cannot be matched to one mesh part. "
                    "Re-import the original model with Import Shape Keys enabled, then repeat your edits."
                )
            obj = objects[0]
            mesh = obj.data
            record_offset = subset_index * MODEL_SUBSET_RECORD_SIZE
            source_index_count = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_INDEX_COUNT_OFFSET)[0]
            source_vertex_count = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_VERTEX_COUNT_OFFSET)[0]
            if len(mesh.vertices) != source_vertex_count:
                raise ValueError(
                    f"{obj.name}'s vertex count changed, so the saved facial shape {target['name']!r} no "
                    "longer fits. Re-import the original model with Import Shape Keys enabled before changing "
                    "the mesh topology."
                )
            subset_base = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_BASE_OFFSET)[0]
            index_offset = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_INDEX_DATA_OFFSET)[0]
            source_indices = list(struct.unpack_from(
                f"<{source_index_count}H",
                template.data,
                geom_base + int(subset_base) + int(index_offset),
            ))
            source_triangles = sorted(
                tuple(sorted(source_indices[index:index + 3]))
                for index in range(0, len(source_indices), 3)
                if len(source_indices[index:index + 3]) == 3
            )
            mesh.calc_loop_triangles()
            current_triangles = sorted(
                tuple(sorted(int(mesh.loops[loop_index].vertex_index) for loop_index in triangle.loops))
                for triangle in mesh.loop_triangles
            )
            if current_triangles != source_triangles:
                raise ValueError(
                    f"{obj.name}'s faces changed, so the saved facial shape {target['name']!r} no longer fits. "
                    "Re-import the original model with Import Shape Keys enabled before changing the mesh faces."
                )
            deltas = {int(index): tuple(value) for index, value in target_subset.get("deltas", {}).items()}
            if any(index < 0 or index >= source_vertex_count for index in deltas):
                raise ValueError(
                    f"The original file has damaged facial-shape data for {target['name']!r}. "
                    "Try a fresh copy of the original extracted .model file."
                )
            result.setdefault(subset_index, []).append({
                "name": str(target["name"]),
                "hash": int(target["hash"]) & U32_MASK,
                "source_index": int(target.get("index", -1)),
                "deltas": deltas,
            })
    if not result:
        raise ValueError(
            "The facial shapes from this older Blender file cannot be matched to the loaded meshes. "
            "Re-import the original model with Import Shape Keys enabled."
        )
    return result, skipped_records



def _build_inert_look_bvh_blocks(template):
    # NOTE: previously wrote ModelLookBVHInfo as a single flat 16-byte
    # all-zero placeholder (struct.pack("<4I", 0, 0, 0, 0)) regardless of how
    # many ModelLook entries the model has. Native SM2 models store one
    # 184-byte record PER look. With raytracing enabled in-game, the engine
    # indexes into ModelLookBVHInfo per look and walks past the undersized
    # 16-byte block into unrelated memory, causing an access violation crash
    # (observed in-game as a crash when RT is toggled on with Luna-exported
    # models; confirmed by three matching crash dumps, all faulting at the
    # same address inside Spider-Man2.exe).
    #
    # Emit one properly-sized record per look instead, using the
    # enabled=1/lod_count=1 "one-LOD" pattern (bytes 0..159 = 0, then
    # u32[160:184] = (1, 1, look_index, 0, 31, 0)) that has been separately
    # verified in-game not to crash. This is a minimal-but-real single-LOD
    # RT state rather than a "disabled" flag: attempts using
    # enabled=0/lod_count=0 for every look were not verified in-game and are
    # not used here.
    replacements = {}
    bvh_hash = BLOCK_HASHES.get("ModelLookBVHInfo")
    if bvh_hash in template.blocks:
        look_payload = template.payload(BLOCK_HASHES["ModelLook"]) if BLOCK_HASHES["ModelLook"] in template.blocks else b""
        look_count = len(look_payload) // MODEL_LOOK_SIZE if look_payload else 0
        look_count = max(look_count, 1)
        records = bytearray(MODEL_BVH_RECORD_SIZE * look_count)
        for look_index in range(look_count):
            struct.pack_into(
                "<6I", records, look_index * MODEL_BVH_RECORD_SIZE + 160,
                1, 1, look_index, 0, 31, 0,
            )
        replacements[bvh_hash] = bytes(records)
    bvh_lod_hash = BLOCK_HASHES.get("ModelLookBVHLoDInfo")
    if bvh_lod_hash in template.blocks:
        replacements[bvh_lod_hash] = b"\x00" * 64
    return replacements


def _block_alignment(block_hash):
    cacheline_blocks = {
        BLOCK_HASHES.get("ModelSubset"),
        BLOCK_HASHES.get("ModelLook"),
        BLOCK_HASHES.get("ModelLookBuilt"),
        BLOCK_HASHES.get("ModelLookBVHInfo"),
    }
    return DAT1_CACHELINE_ALIGN if block_hash in cacheline_blocks else DAT1_BLOCK_ALIGN


def _relocate_model_string_offsets(block_hash, payload, delta):
    if not delta or not payload:
        return payload
    data = bytearray(payload)
    if block_hash == BLOCK_HASHES["ModelMaterial"]:
        material_count = len(data) // (MODEL_MATERIAL_INFO_SIZE + MODEL_MATERIAL_SIZE)
        for index in range(material_count):
            base = index * MODEL_MATERIAL_INFO_SIZE
            for field_offset in (0, 8):
                value = struct.unpack_from("<I", data, base + field_offset)[0]
                if value:
                    struct.pack_into("<I", data, base + field_offset, value + delta)
    elif block_hash == BLOCK_HASHES["ModelLookBuilt"]:
        # Headers occupy the leading fixed-size record array
        look_count = 0
        if len(data) >= MODEL_LOOK_BUILT_SIZE:
            first_section = struct.unpack_from("<Q", data, 0)[0]
            if first_section % MODEL_LOOK_BUILT_SIZE == 0:
                look_count = min(len(data) // MODEL_LOOK_BUILT_SIZE, int(first_section // MODEL_LOOK_BUILT_SIZE))
        for index in range(look_count):
            field_offset = index * MODEL_LOOK_BUILT_SIZE + 76
            value = struct.unpack_from("<I", data, field_offset)[0]
            if value:
                struct.pack_into("<I", data, field_offset, value + delta)
    elif block_hash == BLOCK_HASHES["ModelLookGroup"]:
        group_count = int(data[0]) if data else 0
        for index in range(group_count):
            field_offset = 1 + index * MODEL_LOOK_GROUP_SIZE + 20
            if field_offset + 4 <= len(data):
                value = struct.unpack_from("<I", data, field_offset)[0]
                if value:
                    struct.pack_into("<I", data, field_offset, value + delta)
    return bytes(data)


def _rebuild_dat1(
    template,
    replacements,
    string_pool,
    remove_hashes=None,
    bulk_hash=None,
    bulk_at_end=True,
    pad_bulk_block=True,
    alignment_override=None,
    preserve_original_offsets=False,
):
    if template.fixup_count != 0:
        raise ValueError(
            "This particular game model layout is not supported yet. Try a different original .model file."
        )

    remove_hashes = set(remove_hashes or ())
    physical_hashes = [
        entry[0]
        for entry in sorted(template.entries, key=lambda item: item[1])
        if entry[0] not in remove_hashes
    ]
    geom_hash = int(bulk_hash if bulk_hash is not None else BLOCK_HASHES["ModelSubsetGeomData"])
    added_hashes = [
        block_hash for block_hash in replacements
        if block_hash not in physical_hashes and block_hash not in remove_hashes and block_hash != geom_hash
    ]
    if geom_hash in physical_hashes and bulk_at_end:
        physical_hashes = [h for h in physical_hashes if h != geom_hash] + sorted(added_hashes) + [geom_hash]
    else:
        physical_hashes.extend(sorted(added_hashes))

    strings = string_pool.bytes()
    table_end = DAT1_HEADER_SIZE + len(physical_hashes) * DAT1_BLOCK_TABLE_ENTRY_SIZE + len(template.fixup_table)
    string_base = max(template.sb_offset, table_end)
    string_delta = string_base - template.sb_offset
    cursor = string_base + len(strings)
    payload_by_hash = {}
    offset_by_hash = {}
    body = bytearray()

    for block_hash in physical_hashes:
        payload = replacements.get(block_hash)
        if payload is None:
            payload = template.payload(block_hash)
        if string_delta:
            payload = _relocate_model_string_offsets(block_hash, payload, string_delta)
        if block_hash == geom_hash and pad_bulk_block:
            _original_geom_offset, original_geom_size = template.blocks[geom_hash]
            if len(payload) < original_geom_size:
                payload += b"\x00" * (original_geom_size - len(payload))
        alignment = (
            max(1, int(alignment_override))
            if alignment_override is not None
            else _block_alignment(block_hash)
        )
        aligned = _align(cursor, alignment)
        original_block_offset = int(template.blocks.get(block_hash, (0, 0))[0])
        if (
            original_block_offset
            and aligned <= original_block_offset
            and (preserve_original_offsets or block_hash == geom_hash)
        ):
            aligned = original_block_offset
        pad = aligned - cursor
        if pad:
            body += b"\x00" * pad
            cursor = aligned
        offset_by_hash[block_hash] = cursor
        payload_by_hash[block_hash] = payload
        body += payload
        cursor += len(payload)

    block_table_entries = []
    for block_hash in sorted(payload_by_hash):
        payload = payload_by_hash[block_hash]
        block_table_entries.append(struct.pack("<III", block_hash, offset_by_hash[block_hash], len(payload)))
    block_table = b"".join(block_table_entries)
    # Preserve the source string buffer's absolute DAT1 offset. Many model
    # records store absolute string pointers, including records we otherwise
    # byte-preserve. Removing a block-table entry must therefore leave padding
    # instead of sliding every later payload toward the header.
    header_padding = b"\x00" * max(0, string_base - table_end)
    declared_size = DAT1_HEADER_SIZE + len(block_table) + len(template.fixup_table) + len(header_padding) + len(strings) + len(body)
    header = struct.pack("<IIIHH", DAT1_FILE_ID, template.version, declared_size, len(payload_by_hash), template.fixup_count)
    out = header + block_table + template.fixup_table + header_padding + strings + bytes(body)
    bulk_offset = offset_by_hash.get(geom_hash, 0)
    bulk_size = (
        len(out) - bulk_offset
        if bulk_offset and not bulk_at_end
        else len(payload_by_hash.get(geom_hash, b""))
    )
    return out, bulk_offset, bulk_size


def _asset_chunk_info(size):
    size = int(size)
    if size < 0 or size > ASSET_CHUNK_UNCOMPRESSED_MASK:
        raise ValueError(
            "The exported model is too large for one game file. Split large meshes into smaller objects, "
            "then export again."
        )
    return (
        size
        | (size << ASSET_CHUNK_COMPRESSED_SHIFT)
        | (ASSET_COMPRESSION_NONE << ASSET_CHUNK_COMPRESSION_SHIFT)
    )


def _build_stg_header(version, topology_size, bulk_size):
    chunks = [_asset_chunk_info(topology_size)]
    if bulk_size > 0:
        chunks.append(_asset_chunk_info(bulk_size))
    serialized_header = struct.pack("<IBBH", version, 0, len(chunks), 0)
    serialized_header += b"".join(struct.pack("<Q", chunk) for chunk in chunks)
    stg = bytearray()
    stg += struct.pack("<IIII", STG_MAGIC, STG_VERSION, len(serialized_header), 0)
    stg += serialized_header
    _align_buffer(stg, STG_HEADER_ALIGN)
    return bytes(stg)


def _build_msmr_model_header(template, bulk_offset, bulk_size):
    header = bytearray(template.prefix if len(template.prefix) == 36 else b"\x00" * 36)
    struct.pack_into(
        "<III",
        header,
        0,
        MSMR_MODEL_MAGIC,
        int(bulk_offset),
        int(bulk_size),
    )
    return bytes(header)


def _expected_original_model_name(arm):
    source_name = os.path.basename(_source_path_from_armature(arm))
    if source_name:
        return source_name

    arm_name = str(getattr(arm, "name", "") or "skeleton")
    model_name = re.match(r"^(.*\.model)(?:\.\d{3})?$", arm_name, flags=re.IGNORECASE)
    if model_name:
        return model_name.group(1)
    return f"{arm_name}.model"


class MODEL_OT_select_original_model_for_export(Operator, ImportHelper):
    bl_idname = "model.select_original_model_for_export"
    bl_label = "Please Select Original .model"
    bl_description = "Select the original skeleton model to use as the injection template"
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    stg_mode: StringProperty(default="SCENE", options={'HIDDEN', 'SKIP_SAVE'})
    expected_model_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        expected_name = str(self.expected_model_name or "skeleton.model")
        self.layout.label(text=f"Please select original {expected_name}", icon='ARMATURE_DATA')

    def invoke(self, context, event):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Select the model skeleton (Armature) or one of its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        expected_name = str(self.expected_model_name or _expected_original_model_name(arm))
        self.expected_model_name = expected_name
        missing_path = _source_path_from_armature(arm)
        if missing_path:
            self.filepath = missing_path
        elif not self.filepath:
            self.filepath = expected_name
        self.report(
            {'INFO'},
            f"Please select original {expected_name}.",
        )
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report({'ERROR'}, "The selected model skeleton is no longer available.")
            return {'CANCELLED'}

        source_path = os.path.abspath(self.filepath)
        if not os.path.isfile(source_path):
            self.report({'ERROR'}, "Please select an existing original skeleton .model file.")
            return {'CANCELLED'}

        try:
            template = _Dat1Template(source_path)
        except Exception:
            log_exception("Could not read replacement source model template %s", source_path)
            self.report(
                {'ERROR'},
                "That file is not a readable original game .model. Please select another skeleton model.",
            )
            return {'CANCELLED'}

        arm["engine_model_source_path"] = source_path
        arm["engine_model_source_had_stg"] = bool(template.had_stg)
        source_has_morphs = (
            MSMR_MODEL_ANIM_MORPH_INFO_HASH in template.blocks
            if is_msmr_model(template.blocks)
            else BLOCK_HASHES["ModelAnimMorph2Info"] in template.blocks
        )
        source_has_ziva = BLOCK_HASHES["ModelAnimZiva2Info"] in template.blocks
        arm["engine_model_source_has_morphs"] = source_has_morphs
        arm["engine_model_source_has_ziva"] = source_has_ziva
        if source_has_morphs:
            try:
                source_morph = (
                    decode_msmr_morphs(template.data, template.blocks)
                    if is_msmr_model(template.blocks)
                    else decode_model_morph2(template.data, template.blocks)
                )
                arm["engine_model_morph_target_count"] = int(
                    len((source_morph or {}).get("targets", []))
                )
            except Exception:
                log_exception("Could not count source model shape keys %s", source_path)
        self.report({'INFO'}, f"Using {os.path.basename(source_path)} as the injection model.")
        return bpy.ops.export_scene.engine_model(
            'INVOKE_DEFAULT',
            stg_mode=str(self.stg_mode or "SCENE"),
        )


class ExportEngineModel(Operator, ExportHelper):
    bl_idname = "export_scene.engine_model"
    bl_label = "Export Luna Engine Model"
    bl_description = "Export selected/imported Luna Engine Model geometry using the original .model as a template"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    stg_mode: EnumProperty(
        name="Output Format",
        items=[
            ("SCENE", "Scene Setting", "Use the Luna Engine panel STG checkbox"),
            ("AUTO", "Auto", "Match the imported source file wrapper"),
            ("STG", "STG+DAT1", "Write a model-style STG header before DAT1"),
            ("RAW", "Raw DAT1", "Write DAT1 without an STG header"),
        ],
        default="SCENE",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    def draw(self, context):
        return

    def invoke(self, context, event):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Nothing from an imported model is selected. Select the model skeleton (Armature) or one of "
                "its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        source_path = _source_path_from_armature(arm)
        if not source_path or not os.path.isfile(source_path):
            return bpy.ops.model.select_original_model_for_export(
                'INVOKE_DEFAULT',
                stg_mode=str(self.stg_mode or "SCENE"),
                expected_model_name=_expected_original_model_name(arm),
            )
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Nothing from an imported model is selected. Select the model skeleton (Armature) or one of "
                "its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        source_path = _source_path_from_armature(arm)
        if not source_path or not os.path.isfile(source_path):
            self.report(
                {'ERROR'},
                "I can't find the original .model file used by this Blender scene. Import the original model "
                "again, then export.",
            )
            return {'CANCELLED'}

        try:
            template = _Dat1Template(source_path)
        except Exception as exc:
            log_exception("Could not read source model template %s", source_path)
            self.report(
                {'ERROR'},
                "I couldn't open the original .model file. Make sure it still exists and is an original "
                "extracted game model, then import it again.",
            )
            return {'CANCELLED'}

        msmr = is_msmr_model(template.blocks)
        source_has_morph = (
            MSMR_MODEL_ANIM_MORPH_INFO_HASH in template.blocks
            if msmr
            else BLOCK_HASHES["ModelAnimMorph2Info"] in template.blocks
        )
        source_has_ziva = not msmr and BLOCK_HASHES["ModelAnimZiva2Info"] in template.blocks
        source_has_smooth = BLOCK_HASHES["ModelAnimVertSmoothInfo"] in template.blocks
        discard_unimported_morphs = bool(getattr(arm, "engine_model_discard_unimported_morphs", False))
        compact_lod0 = bool(
            msmr
            and getattr(arm, "engine_model_compact_lod0_export", False)
            and not bool(arm.get("engine_model_import_all_lods", False))
        )
        recover_source_morph = (
            source_has_morph
            and not msmr
            and not bool(arm.get("engine_model_shape_keys_imported", False))
            and not discard_unimported_morphs
        )

        required = (
            ("ModelBuilt", "ModelMaterial", "ModelLook", "ModelLookGroup", "ModelLookBuilt", "ModelSubset", "ModelIndex", "ModelStdVert")
            if msmr
            else ("ModelBuilt", "ModelMaterial", "ModelLook", "ModelLookGroup", "ModelLookBuilt", "ModelSubset", "ModelSubsetGeomData")
        )
        missing = [name for name in required if BLOCK_HASHES[name] not in template.blocks]
        if missing:
            self.report(
                {'ERROR'},
                "The chosen source file is not a complete game model. Import a different original .model "
                f"file and try again. Missing internal data: {', '.join(missing)}.",
            )
            return {'CANCELLED'}

        mesh_objects = [
            obj for obj in bpy.data.objects
            if getattr(obj, "type", None) == "MESH" and getattr(obj, "parent", None) == arm
            and obj.get("engine_bounds_type", "") != "subset_aabb"
        ]
        resolve_subset_index_collisions(arm)
        # A LOD0-only import intentionally has no Blender objects for lower
        # source LODs. Do not treat their look subset IDs as stale unless the
        # user has explicitly switched to editing the look definitions.
        if (
            bool(arm.get("engine_model_import_all_lods", False))
            or bool(arm.get("engine_model_looks_modified", False))
        ):
            sanitize_model_look_metadata(arm, mark_modified=True)

        def subset_sort_key(obj):
            value = obj.get("engine_subset_index", None)
            return (999999 if value is None else int(value), obj.name)

        mesh_objects.sort(key=subset_sort_key)
        if not mesh_objects:
            self.report(
                {'ERROR'},
                "No model meshes were found under the selected skeleton. Parent at least one mesh directly "
                "to the Armature, then export again.",
            )
            return {'CANCELLED'}

        if msmr:
            wm = context.window_manager
            wm.progress_begin(0, 100)
            export_warnings = []
            try:
                wm.progress_update(10)
                original_materials = _parse_model_materials(template.data, template.blocks)
                material_entries, material_indices = _build_material_entries(
                    mesh_objects,
                    original_materials,
                    export_warnings,
                )
                string_pool = _StringPool(template.sb_offset, template.string_buffer)
                hierarchy_hash = BLOCK_HASHES["ModelJointHierarchy"]
                if hierarchy_hash in template.blocks:
                    hierarchy = template.payload(hierarchy_hash)
                    source_joint_count = struct.unpack_from("<H", hierarchy, 2)[0] if len(hierarchy) >= 4 else 0
                else:
                    source_joint_count = 0

                source_morph_targets_by_subset = {}
                recover_msmr_morphs = (
                    source_has_morph
                    and not bool(arm.get("engine_model_shape_keys_imported", False))
                    and not discard_unimported_morphs
                )
                if recover_msmr_morphs:
                    source_morph = decode_msmr_morphs(template.data, template.blocks)
                    for target_index, target in enumerate((source_morph or {}).get("targets", [])):
                        for subset in target.get("subsets", []):
                            source_morph_targets_by_subset.setdefault(
                                int(subset.get("subset_index", -1)),
                                [],
                            ).append({
                                "name": str(target.get("name", f"Morph_{target_index}")),
                                "hash": int(target.get("hash", 0)) & U32_MASK,
                                "source_index": int(target.get("index", target_index)),
                                "deltas": dict(subset.get("deltas", {})),
                            })

                wm.progress_update(30)
                geometry = _build_msmr_geometry_blocks(
                    mesh_objects,
                    arm,
                    material_indices,
                    template,
                    source_joint_count,
                    export_warnings=export_warnings,
                    source_morph_targets_by_subset=source_morph_targets_by_subset,
                    compact_lod0=compact_lod0,
                )
                replacements = dict(geometry["replacements"])
                replacements[BLOCK_HASHES["ModelMaterial"]] = _build_material_block(
                    material_entries,
                    string_pool,
                )
                source_subset_count = (
                    len(template.payload(BLOCK_HASHES["ModelSubset"]))
                    // MSMR_SUBSET_RECORD_SIZE
                )
                source_subset_layout_preserved = (
                    bool(geometry.get("source_subset_layout_preserved", False))
                    or (
                        len(geometry["stats"]) == source_subset_count
                        and all(
                            int(stat.get("source_subset_index", -1)) == subset_index
                            for subset_index, stat in enumerate(geometry["stats"])
                        )
                    )
                )
                looks_modified = bool(arm.get("engine_model_looks_modified", False))
                stale_lod0_look_edit = (
                    looks_modified
                    and not bool(arm.get("engine_model_import_all_lods", False))
                    and _msmr_lod0_scene_looks_match_source(arm, template)
                )
                preserve_msmr_looks = (
                    source_subset_layout_preserved
                    and (not looks_modified or stale_lod0_look_edit)
                )
                if compact_lod0:
                    look_block = _build_msmr_compact_lod0_look_block(
                        template,
                        len(geometry["stats"]),
                    )
                    look_built_block = _build_msmr_compact_lod0_look_built_block(
                        template,
                        look_block,
                    )
                    look_group_block = template.payload(BLOCK_HASHES["ModelLookGroup"])
                elif preserve_msmr_looks:
                    look_block = template.payload(BLOCK_HASHES["ModelLook"])
                    look_built_block = template.payload(BLOCK_HASHES["ModelLookBuilt"])
                    look_group_block = template.payload(BLOCK_HASHES["ModelLookGroup"])
                else:
                    look_block, look_built_block, look_group_block = _build_msmr_look_blocks(
                        len(geometry["stats"]),
                        string_pool,
                        template,
                        arm,
                        geometry["subset_index_map"],
                    )
                replacements[BLOCK_HASHES["ModelLook"]] = look_block
                replacements[BLOCK_HASHES["ModelLookBuilt"]] = look_built_block
                replacements[BLOCK_HASHES["ModelLookGroup"]] = look_group_block
                replacements[BLOCK_HASHES["ModelBuilt"]] = _build_msmr_model_built_block(
                    template,
                    geometry,
                )

                remove_hashes = set()
                morph_count = 0
                if source_has_morph and bool(geometry.get("preserve_source_morphs", False)):
                    replacements.update({
                        MSMR_MODEL_ANIM_MORPH_INFO_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_INFO_HASH),
                        MSMR_MODEL_ANIM_MORPH_DATA_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_DATA_HASH),
                        MSMR_MODEL_ANIM_MORPH_INDICES_HASH: template.payload(MSMR_MODEL_ANIM_MORPH_INDICES_HASH),
                    })
                    source_morph = decode_msmr_morphs(template.data, template.blocks)
                    morph_count = int((source_morph or {}).get("target_count", 0))
                elif source_has_morph and (
                    bool(arm.get("engine_model_shape_keys_imported", False))
                    or recover_msmr_morphs
                ):
                    morph_result = _build_msmr_morph_blocks(
                        template,
                        geometry["morph_targets"],
                        string_pool,
                    )
                    if morph_result is not None:
                        morph_replacements, morph_count = morph_result
                        replacements.update(morph_replacements)
                elif source_has_morph and discard_unimported_morphs:
                    remove_hashes.update({
                        MSMR_MODEL_ANIM_MORPH_INFO_HASH,
                        MSMR_MODEL_ANIM_MORPH_DATA_HASH,
                        MSMR_MODEL_ANIM_MORPH_INDICES_HASH,
                    })
                    export_warnings.append(
                        "Facial animation was left out because 'Discard Unimported Morphs' is turned on."
                    )

                preserve_msmr_bvh = (
                    preserve_msmr_looks
                    and bool(geometry.get("source_geometry_unchanged", False))
                )
                bvh_hash = BLOCK_HASHES.get("ModelLookBVHInfo")
                if bvh_hash in template.blocks and not preserve_msmr_bvh:
                    replacements[bvh_hash] = b"\x00" * template.blocks[bvh_hash][1]

                wm.progress_update(75)
                dat1, bulk_offset, bulk_size = _rebuild_dat1(
                    template,
                    replacements,
                    string_pool,
                    remove_hashes=remove_hashes,
                    bulk_hash=MSMR_MODEL_INDEX_HASH,
                    bulk_at_end=False,
                    pad_bulk_block=False,
                    alignment_override=DAT1_BLOCK_ALIGN,
                    preserve_original_offsets=True,
                )
                stg_mode = str(getattr(self, "stg_mode", "SCENE") or "SCENE")
                source_had_wrapper = len(template.prefix) == 36
                if stg_mode == "STG":
                    add_wrapper = True
                elif stg_mode == "RAW":
                    add_wrapper = False
                else:
                    # AUTO and the file-menu default both preserve MSMR's native
                    # wrapper. The scene's STG checkbox belongs to MSM2 and must
                    # not silently turn an MSMR .model into raw DAT1.
                    add_wrapper = source_had_wrapper
                if add_wrapper:
                    out = _build_msmr_model_header(template, bulk_offset, bulk_size) + dat1
                    format_name = "MSMR model wrapper+DAT1"
                else:
                    out = dat1
                    format_name = "raw DAT1"
                with open(self.filepath, "wb") as file:
                    file.write(out)
            except Exception as exc:
                log_exception("MSMR model export failed")
                wm.progress_end()
                self.report({'ERROR'}, f"Export couldn't finish. {_friendly_export_error(exc)}")
                return {'CANCELLED'}

            wm.progress_update(100)
            wm.progress_end()
            if export_warnings:
                for warning in export_warnings:
                    log_warning("MSMR model export check: %s", warning)
                self.report({'WARNING'}, _format_export_warnings(export_warnings))
            self.report(
                {'INFO'},
                f"Export finished ({format_name}): {len(geometry['stats'])} mesh part(s), "
                f"{geometry.get('live_vertex_count', geometry['vertex_count'])} vertices, "
                f"{geometry.get('live_index_count', geometry['index_count']) // 3} triangles, and "
                f"{morph_count} facial shape(s).",
            )
            return {'FINISHED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        export_warnings = []
        try:
            wm.progress_update(10)
            original_materials = _parse_model_materials(template.data, template.blocks)
            material_entries, material_indices = _build_material_entries(mesh_objects, original_materials, export_warnings)
            string_pool = _StringPool(template.sb_offset, template.string_buffer)

            wm.progress_update(30)
            hierarchy_hash = BLOCK_HASHES["ModelJointHierarchy"]
            if hierarchy_hash in template.blocks:
                hierarchy = template.payload(hierarchy_hash)
                source_joint_count = struct.unpack_from("<H", hierarchy, 2)[0] if len(hierarchy) >= 4 else None
            else:
                source_joint_count = 0
            source_morph_targets_by_subset = {}
            if recover_source_morph:
                try:
                    source_morph_targets_by_subset, skipped_morph_subset_records = _source_morph_targets_for_older_scene(
                        template,
                        mesh_objects,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{exc} If you do not need facial animation, turn on 'Discard Unimported Morphs' "
                        "under Model > Export and try again."
                    ) from exc
                log_debug(
                    "Older scene had no imported shape keys; compatible source facial shapes were recovered."
                )
                if skipped_morph_subset_records:
                    export_warnings.append(
                        f"{skipped_morph_subset_records} facial-shape part(s) could not be matched because their "
                        "meshes are not loaded. Re-import with Import All LODs and Import Shape Keys enabled if "
                        "you need those parts."
                    )
            elif source_has_morph and discard_unimported_morphs:
                export_warnings.append(
                    "Facial animation was left out because 'Discard Unimported Morphs' is turned on. Turn it "
                    "off and re-import with Import Shape Keys enabled if you want facial animation."
                )

            subset_block, geom_block, subset_stats, subset_index_map, morph_targets = _build_geometry_and_subset_blocks(
                mesh_objects,
                arm,
                material_indices,
                template,
                source_joint_count,
                export_warnings=export_warnings,
                source_morph_targets_by_subset=source_morph_targets_by_subset,
            )
            morph_info_block, morph_geom_suffix, morph_metadata = encode_model_morph2(
                morph_targets,
                len(geom_block),
            )
            has_morph = morph_info_block is not None
            if has_morph:
                geom_block += morph_geom_suffix
            material_block = _build_material_block(material_entries, string_pool)
            generated_subset_count = len(subset_stats)
            look_block, look_built_block, look_group_block = _build_look_blocks(
                generated_subset_count,
                string_pool,
                template,
                arm=arm,
                subset_index_map=subset_index_map,
            )
            has_smooth = bool(has_morph and source_has_smooth and not source_has_ziva)
            model_built_block = _build_model_built_block(
                template,
                subset_stats,
                arm=arm,
                has_morph=has_morph,
                has_smooth=has_smooth,
            )

            replacements = {
                BLOCK_HASHES["ModelBuilt"]: model_built_block,
                BLOCK_HASHES["ModelMaterial"]: material_block,
                BLOCK_HASHES["ModelLook"]: look_block,
                BLOCK_HASHES["ModelLookBuilt"]: look_built_block,
                BLOCK_HASHES["ModelLookGroup"]: look_group_block,
                BLOCK_HASHES["ModelSubset"]: subset_block,
                BLOCK_HASHES["ModelSubsetGeomData"]: geom_block,
            }
            if arm is not None and hair_objects_for_armature(arm):
                hair_blocks, hair_warnings = compile_export_hair(arm)
                if hair_blocks:
                    replacements.update(hair_blocks)
                    # ModelBuilt.m_StrandSubsetCount cant be zero
                    subset_count = len(hair_blocks[BLOCK_HASHES["ModelStrandSubsets"]]) // SUBSET_SIZE
                    model_built = bytearray(replacements[BLOCK_HASHES["ModelBuilt"]])
                    struct.pack_into(
                        "<b", model_built, MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET,
                        max(0, min(127, subset_count)),
                    )
                    replacements[BLOCK_HASHES["ModelBuilt"]] = bytes(model_built)
                export_warnings.extend(hair_warnings)
            remove_hashes = set()
            if has_morph:
                replacements[BLOCK_HASHES["ModelAnimVertInfo2"]] = b"\x00" * 40
                replacements[BLOCK_HASHES["ModelAnimMorph2Info"]] = morph_info_block
                if has_smooth:
                    smooth_info_block, smooth_metadata = encode_model_smooth2(subset_stats)
                    replacements[BLOCK_HASHES["ModelAnimVertSmoothInfo"]] = smooth_info_block
                    cross_subset_unstitched = smooth_metadata.get("cross_subset_unstitched_count", 0)
                    if cross_subset_unstitched > 0:
                        export_warnings.append(
                            f"{cross_subset_unstitched} seam vertex(es) at subset/material boundaries "
                            "coincide with a vertex in a different subset and could not be included in "
                            "any normal-smoothing stitch (the file format only supports stitches within "
                            "a single subset). Shading/tangents may show a visible seam there under strong "
                            "shape-key deformation."
                        )
                else:
                    remove_hashes.add(BLOCK_HASHES["ModelAnimVertSmoothInfo"])
                if source_has_ziva:
                    remove_hashes.add(BLOCK_HASHES["ModelAnimZiva2Info"])
                    log_debug(
                        "Converted registered shape keys from Ziva to Morph2 while preserving source shading."
                    )
            elif source_has_morph:
                remove_hashes.update({
                    BLOCK_HASHES["ModelAnimMorph2Info"],
                    BLOCK_HASHES["ModelAnimVertSmoothInfo"],
                    BLOCK_HASHES["ModelAnimVertInfo2"],
                })
            elif source_has_ziva:
                remove_hashes.update({
                    BLOCK_HASHES["ModelAnimZiva2Info"],
                    BLOCK_HASHES["ModelAnimVertSmoothInfo"],
                    BLOCK_HASHES["ModelAnimVertInfo2"],
                })
            replacements.update(_build_inert_look_bvh_blocks(template))

            wm.progress_update(75)
            dat1, geom_offset, geom_size = _rebuild_dat1(
                template,
                replacements,
                string_pool,
                remove_hashes=remove_hashes,
            )
            source_had_stg = bool(template.had_stg or arm.get("engine_model_source_had_stg", False))
            stg_mode = str(getattr(self, "stg_mode", "SCENE") or "SCENE")
            if stg_mode == "STG":
                add_stg = True
            elif stg_mode == "RAW":
                add_stg = False
            elif stg_mode == "AUTO":
                add_stg = source_had_stg
            else:
                add_stg = bool(getattr(context.scene, "engine_export_add_stg_header", True))
            if add_stg:
                stg_header = _build_stg_header(template.version, geom_offset, geom_size)
                out = stg_header + dat1
                format_name = "STG+DAT1"
            else:
                out = dat1
                format_name = "raw DAT1"

            with open(self.filepath, "wb") as f:
                f.write(out)
        except Exception as exc:
            log_exception("Model export failed")
            wm.progress_end()
            self.report({'ERROR'}, f"Export couldn't finish. {_friendly_export_error(exc)}")
            return {'CANCELLED'}

        wm.progress_update(100)
        wm.progress_end()
        if export_warnings:
            for warning in export_warnings:
                log_warning("Model export check: %s", warning)
            self.report({'WARNING'}, _format_export_warnings(export_warnings))
        self.report(
            {'INFO'},
            f"Export finished ({format_name}): {len(mesh_objects)} mesh part(s), "
            f"{sum(s['vertex_count'] for s in subset_stats)} vertices, "
            f"{sum(s['index_count'] // 3 for s in subset_stats)} triangles, and "
            f"{len(morph_metadata.get('targets', [])) if has_morph else 0} facial shape(s)."
        )
        return {'FINISHED'}
