#Strands

from .utils import *

import math

STRAND_SIZE = 12
CV_SIZE = 8
SKIN_BINDING_SIZE = 8
JOINT_BINDING_SIZE = 16
CV_WEIGHTS_SIZE = 8
SUBSET_SIZE = 1256
DESCRIPTION_SIZE = 1112
MAX_STRAND_JOINT_WEIGHTS = 12

SUBSET_NAME_HASH_OFF = 0
SUBSET_NAME_OFFSET_OFF = 4
SUBSET_STRAND_COUNT_OFF = 8
SUBSET_STRAND_START_OFF = 12
SUBSET_DESCRIPTION_OFF = 16
SUBSET_SKINNING_PACK_OFF = 1172

# ModelStrandDescription field offsets 
DESC_LOD_DISTANCE = 0
DESC_LOD_REDUCTION = 4
DESC_TESS_MAX = 8
DESC_TESS_MIN = 10
DESC_CLUMP_MATERIAL_PACK = 12  # StrandsPerClumpMax:14 | StrandsPerClumpMin:14 | MaterialType:4
DESC_STRAND_ROUNDNESS = 16
DESC_STRAND_LENGTH_RANDOM = 20
DESC_STRAND_THICKNESS_RANDOM = 24
DESC_CLUMP_ROUNDNESS = 28
DESC_GLOSS = 64
DESC_BOOL_PACK = 100  # SimulationEnable:1 | SkipShadows:1 | CurlsEnable:1 | SkipRendering:1 | Reserved:28
DESC_SIM_STIFFNESS_LENGTH = 104
DESC_SIM_STIFFNESS_POWER = 108
DESC_SIM_DRAG = 112

DESC_DIFFUSE_ENVELOPE_OFF = 248
DESC_ENVELOPE_POINT_COUNT = 4
DESC_ENVELOPE_POINT_SIZE = 16

HAIR_DESCRIPTION_KEY = "engine_hair_description_b64"

MODEL_BUILT_COMMON_MPU_OFFSET = 60
MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET = 78

SKINNING_DISABLED, SKINNING_GEOMETRY, SKINNING_JOINTS = 0, 1, 2
SKINNING_NAMES = {SKINNING_DISABLED: "disabled", SKINNING_GEOMETRY: "geometry", SKINNING_JOINTS: "joints"}
SKINNING_IDS = {v: k for k, v in SKINNING_NAMES.items()}

HAIR_SUBSET_KEY = "engine_hair_subset_name"
HAIR_SKINNING_KEY = "engine_hair_skinning_type"
HAIR_SOURCE_KEY = "engine_hair_source"

STRAND_BLOCK_NAMES = (
    "ModelStrandSubsets",
    "ModelStrands",
    "ModelStrandCVs",
    "ModelStrandSBs",
    "ModelStrandJBs",
    "ModelStrandCVWeights",
)


# taken from Core/Foundation/Math/Vec.cpp


def _vec_to_i16(vec, scale):
    return tuple(max(-32768, min(32767, round(component * scale))) for component in vec)


def _vec_from_i16(values, inv_scale):
    return tuple(component * inv_scale for component in values)


def _vec_normalize(vec):
    length = math.sqrt(sum(component * component for component in vec))
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vec)


def _encode_octahedron(vec):
    x, y, z = _vec_normalize(vec)
    denom = max(abs(x) + abs(y) + abs(z), 1e-6)  # L1 norm, not max(|x|,|y|,|z|)
    x, y, z = x / denom, y / denom, z / denom
    if y >= 0:
        y = z
    else:
        sign_x = 1.0 if x >= 0 else -1.0
        sign_z = 1.0 if z >= 0 else -1.0
        new_x = (1.0 - abs(z)) * sign_x
        new_y = (1.0 - abs(x)) * sign_z
        x, y = new_x, new_y
    return (x * 0.5 + 0.5, y * 0.5 + 0.5)


def _decode_octahedron(uv):
    ex, ey = uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0
    x, y, z = ex, 1.0 - abs(ex) - abs(ey), ey
    t = max(0.0, min(1.0, -y))
    x += -t if x >= 0.0 else t
    z += -t if z >= 0.0 else t
    return (x, y, z)


def _pack_normal_bytes(vec):
    u, v = _encode_octahedron(vec)
    return (
        max(0, min(255, int(u * 255.0 + 0.5))),
        max(0, min(255, int(v * 255.0 + 0.5))),
    )


def _unpack_normal_bytes(byte_pair):
    return _decode_octahedron((byte_pair[0] / 255.0, byte_pair[1] / 255.0))




def _pack_strand(cv_start, cv_count, uv_raw, basis_u_bytes, basis_v_bytes):
    if not (0 <= cv_start < (1 << 24)) or not (0 <= cv_count < 256):
        raise ValueError(f"Strand point range ({cv_start}, {cv_count}) is outside the supported format limits.")
    header = (cv_start & 0xFFFFFF) | ((cv_count & 0xFF) << 24)
    return struct.pack(
        "<I2H2B2B",
        header,
        *(max(0, min(65535, round(value))) for value in uv_raw),
        *basis_u_bytes,
        *basis_v_bytes,
    )


def _unpack_strand(data, offset):
    header, uv0, uv1, bu0, bu1, bv0, bv1 = struct.unpack_from("<I2H2B2B", data, offset)
    return {
        "cv_start": header & 0xFFFFFF,
        "cv_count": (header >> 24) & 0xFF,
        "uv": (uv0, uv1),
        "basis_u_bytes": (bu0, bu1),
        "basis_v_bytes": (bv0, bv1),
    }


def _pack_cv(position_i16, normal_bytes):
    return struct.pack("<3h2B", *position_i16, *normal_bytes)


def _unpack_cv(data, offset):
    x, y, z, nx, ny = struct.unpack_from("<3h2B", data, offset)
    return (x, y, z), (nx, ny)


def _pack_skin_binding(v0, v1, v2, w0, w1, subset_index):
    word0 = (v0 & 0xFFFF) | ((v1 & 0xFFFF) << 16)
    word1 = (v2 & 0xFFFF) | ((w0 & 0xF) << 16) | ((w1 & 0xF) << 20) | ((subset_index & 0xFF) << 24)
    return struct.pack("<II", word0, word1)


def _unpack_skin_binding(data, offset):
    word0, word1 = struct.unpack_from("<II", data, offset)
    return {
        "vertex_indices": (word0 & 0xFFFF, (word0 >> 16) & 0xFFFF, word1 & 0xFFFF),
        "weight0": (word1 >> 16) & 0xF,
        "weight1": (word1 >> 20) & 0xF,
        "subset_index": (word1 >> 24) & 0xFF,
    }


def _pack_joint_binding(joint_indices):
    padded = list(joint_indices[:MAX_STRAND_JOINT_WEIGHTS]) + [0x3FF] * (
        MAX_STRAND_JOINT_WEIGHTS - len(joint_indices)
    )
    words = []
    for group in range(4):
        a, b, c = padded[group * 3 : group * 3 + 3]
        for value in (a, b, c):
            if not 0 <= value < 1024:
                raise ValueError(f"Bone index {value} is outside the supported range.")
        words.append((a & 0x3FF) | ((b & 0x3FF) << 10) | ((c & 0x3FF) << 20))
    return struct.pack("<4I", *words)


def _unpack_joint_binding(data, offset):
    words = struct.unpack_from("<4I", data, offset)
    indices = []
    for word in words:
        indices.append(word & 0x3FF)
        indices.append((word >> 10) & 0x3FF)
        indices.append((word >> 20) & 0x3FF)
    return indices


# First 5 slots (7+7+6+6+6=32 bits) pack word0
_WEIGHT_WORD_LAYOUT = (
    (0, ((0, 7), (7, 7), (14, 6), (20, 6), (26, 6))),
    (1, ((0, 5), (5, 5), (10, 5), (15, 5), (20, 4), (24, 4), (28, 4))),
)


def _pack_cv_weights(fractions):
    values = list(fractions[:MAX_STRAND_JOINT_WEIGHTS]) + [0.0] * (
        MAX_STRAND_JOINT_WEIGHTS - len(fractions)
    )
    words = [0, 0]
    slot = 0
    for word_index, layout in _WEIGHT_WORD_LAYOUT:
        for shift, bits in layout:
            maximum = (1 << bits) - 1
            quantized = max(0, min(maximum, int(values[slot] * maximum)))
            words[word_index] |= quantized << shift
            slot += 1
    return struct.pack("<2I", *words)


def _unpack_cv_weights(data, offset):
    word0, word1 = struct.unpack_from("<2I", data, offset)
    words = (word0, word1)
    fractions = []
    for word_index, layout in _WEIGHT_WORD_LAYOUT:
        word = words[word_index]
        for shift, bits in layout:
            maximum = (1 << bits) - 1
            quantized = (word >> shift) & maximum
            fractions.append(quantized / maximum if maximum else 0.0)
    return fractions


def _quantize_strand_weights(per_cv_raw_weights):
    if not per_cv_raw_weights:
        return [], []
    joint_count = len(per_cv_raw_weights[0])
    totals = [sum(cv[joint] for cv in per_cv_raw_weights) for joint in range(joint_count)]
    order = sorted(range(joint_count), key=lambda index: -totals[index])
    kept = order[:MAX_STRAND_JOINT_WEIGHTS]
    result = []
    for cv in per_cv_raw_weights:
        row = [cv[index] for index in kept]
        row = row + [0.0] * (MAX_STRAND_JOINT_WEIGHTS - len(row))
        for slot in range(5, MAX_STRAND_JOINT_WEIGHTS):
            row[slot] = math.sqrt(max(0.0, row[slot]))
        result.append(row)
    return result, kept


def _sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)



DEFAULT_DESCRIPTION = bytearray(DESCRIPTION_SIZE)
struct.pack_into("<f", DEFAULT_DESCRIPTION, DESC_LOD_DISTANCE, 30.0)
struct.pack_into("<f", DEFAULT_DESCRIPTION, DESC_LOD_REDUCTION, 0.5)
struct.pack_into("<H", DEFAULT_DESCRIPTION, DESC_TESS_MAX, 8)
struct.pack_into("<H", DEFAULT_DESCRIPTION, DESC_TESS_MIN, 2)
struct.pack_into("<I", DEFAULT_DESCRIPTION, DESC_CLUMP_MATERIAL_PACK, (1 & 0x3FFF) | ((1 & 0x3FFF) << 14))
struct.pack_into("<f", DEFAULT_DESCRIPTION, DESC_STRAND_ROUNDNESS, 1.0)
struct.pack_into("<f", DEFAULT_DESCRIPTION, DESC_GLOSS, 0.3)
DEFAULT_DESCRIPTION = bytes(DEFAULT_DESCRIPTION)


def _description_get_editable(raw):
    if not raw or len(raw) < DESCRIPTION_SIZE:
        raw = DEFAULT_DESCRIPTION
    clump_pack = struct.unpack_from("<I", raw, DESC_CLUMP_MATERIAL_PACK)[0]
    bool_pack = struct.unpack_from("<I", raw, DESC_BOOL_PACK)[0]
    return {
        "lod_distance": struct.unpack_from("<f", raw, DESC_LOD_DISTANCE)[0],
        "lod_reduction": struct.unpack_from("<f", raw, DESC_LOD_REDUCTION)[0],
        "tess_max": struct.unpack_from("<H", raw, DESC_TESS_MAX)[0],
        "tess_min": struct.unpack_from("<H", raw, DESC_TESS_MIN)[0],
        "strands_per_clump_max": clump_pack & 0x3FFF,
        "strands_per_clump_min": (clump_pack >> 14) & 0x3FFF,
        "strand_roundness": struct.unpack_from("<f", raw, DESC_STRAND_ROUNDNESS)[0],
        "strand_length_random": struct.unpack_from("<f", raw, DESC_STRAND_LENGTH_RANDOM)[0],
        "strand_thickness_random": struct.unpack_from("<f", raw, DESC_STRAND_THICKNESS_RANDOM)[0],
        "clump_roundness": struct.unpack_from("<f", raw, DESC_CLUMP_ROUNDNESS)[0],
        "gloss": struct.unpack_from("<f", raw, DESC_GLOSS)[0],
        "simulation_enable": bool(bool_pack & 0x1),
        "curls_enable": bool(bool_pack & 0x4),
        "simulation_stiffness_length": struct.unpack_from("<f", raw, DESC_SIM_STIFFNESS_LENGTH)[0],
        "simulation_stiffness_power": struct.unpack_from("<f", raw, DESC_SIM_STIFFNESS_POWER)[0],
        "simulation_drag": struct.unpack_from("<f", raw, DESC_SIM_DRAG)[0],
        "color_points": [
            struct.unpack_from("<3f", raw, DESC_DIFFUSE_ENVELOPE_OFF + i * DESC_ENVELOPE_POINT_SIZE)
            for i in range(DESC_ENVELOPE_POINT_COUNT)
        ],
    }


def _description_apply_editable(raw, values):
    out = bytearray(raw) if raw and len(raw) >= DESCRIPTION_SIZE else bytearray(DEFAULT_DESCRIPTION)
    struct.pack_into("<f", out, DESC_LOD_DISTANCE, float(values["lod_distance"]))
    struct.pack_into("<f", out, DESC_LOD_REDUCTION, float(values["lod_reduction"]))
    struct.pack_into("<H", out, DESC_TESS_MAX, max(0, min(65535, int(values["tess_max"]))))
    struct.pack_into("<H", out, DESC_TESS_MIN, max(0, min(65535, int(values["tess_min"]))))
    material_type = struct.unpack_from("<I", out, DESC_CLUMP_MATERIAL_PACK)[0] >> 28
    clump_pack = (
        (max(0, min(0x3FFF, int(values["strands_per_clump_max"]))))
        | (max(0, min(0x3FFF, int(values["strands_per_clump_min"]))) << 14)
        | (material_type << 28)
    )
    struct.pack_into("<I", out, DESC_CLUMP_MATERIAL_PACK, clump_pack)
    struct.pack_into("<f", out, DESC_STRAND_ROUNDNESS, float(values["strand_roundness"]))
    struct.pack_into("<f", out, DESC_STRAND_LENGTH_RANDOM, float(values["strand_length_random"]))
    struct.pack_into("<f", out, DESC_STRAND_THICKNESS_RANDOM, float(values["strand_thickness_random"]))
    struct.pack_into("<f", out, DESC_CLUMP_ROUNDNESS, float(values["clump_roundness"]))
    struct.pack_into("<f", out, DESC_GLOSS, float(values["gloss"]))
    # m
    bool_pack = struct.unpack_from("<I", out, DESC_BOOL_PACK)[0] & ~0x5
    bool_pack |= (1 if values["simulation_enable"] else 0)
    bool_pack |= (4 if values["curls_enable"] else 0)
    struct.pack_into("<I", out, DESC_BOOL_PACK, bool_pack)
    struct.pack_into("<f", out, DESC_SIM_STIFFNESS_LENGTH, float(values["simulation_stiffness_length"]))
    struct.pack_into("<f", out, DESC_SIM_STIFFNESS_POWER, float(values["simulation_stiffness_power"]))
    struct.pack_into("<f", out, DESC_SIM_DRAG, float(values["simulation_drag"]))
    color_points = values.get("color_points")
    if color_points:
        for i in range(DESC_ENVELOPE_POINT_COUNT):
            point_off = DESC_DIFFUSE_ENVELOPE_OFF + i * DESC_ENVELOPE_POINT_SIZE
            r, g, b = color_points[i]
            struct.pack_into("<3f", out, point_off, float(r), float(g), float(b))
    return bytes(out)


class MODEL_PG_hair_description(PropertyGroup):
    lod_distance: FloatProperty(name="LOD Distance", default=30.0, min=0.0, soft_max=500.0)
    lod_reduction: FloatProperty(name="LOD Reduction", default=0.5, min=0.0, max=1.0, subtype='FACTOR')
    tess_max: IntProperty(name="Tessellation Max", default=8, min=0, soft_max=64)
    tess_min: IntProperty(name="Tessellation Min", default=2, min=0, soft_max=64)
    clump_max: IntProperty(name="Strands Per Clump Max", default=1, min=0, soft_max=4095)
    clump_min: IntProperty(name="Strands Per Clump Min", default=1, min=0, soft_max=4095)
    strand_roundness: FloatProperty(name="Strand Roundness", default=1.0, min=0.0, max=1.0, subtype='FACTOR')
    strand_length_random: FloatProperty(name="Length Random", default=0.0, min=0.0, max=1.0, subtype='FACTOR')
    strand_thickness_random: FloatProperty(name="Thickness Random", default=0.0, min=0.0, max=1.0, subtype='FACTOR')
    clump_roundness: FloatProperty(name="Clump Roundness", default=0.0, min=0.0, max=1.0, subtype='FACTOR')
    gloss: FloatProperty(name="Gloss", default=0.3, min=0.0, max=1.0, subtype='FACTOR')
    simulation_enable: BoolProperty(name="Simulation Enabled", default=True)
    curls_enable: BoolProperty(name="Curls Enabled", default=False)
    sim_stiffness_length: FloatProperty(name="Sim Stiffness (Length)", default=0.0, min=0.0, max=1.0, subtype='FACTOR')
    sim_stiffness_power: FloatProperty(name="Sim Stiffness (Power)", default=0.0, min=0.0, soft_max=8.0)
    sim_drag: FloatProperty(name="Sim Drag", default=0.0, min=0.0, max=1.0, subtype='FACTOR')

    color_simple: FloatVectorProperty(
        name="Color", description="A flat color to apply across the whole strand (see the button below)",
        subtype='COLOR', size=3, min=0.0, soft_max=1.0, default=(1.0, 1.0, 1.0),
    )
    color_root: FloatVectorProperty(name="Root", subtype='COLOR', size=3, min=0.0, soft_max=1.0, default=(1.0, 1.0, 1.0))
    color_root_third: FloatVectorProperty(name="Root-Third", subtype='COLOR', size=3, min=0.0, soft_max=1.0, default=(1.0, 1.0, 1.0))
    color_tip_third: FloatVectorProperty(name="Tip-Third", subtype='COLOR', size=3, min=0.0, soft_max=1.0, default=(1.0, 1.0, 1.0))
    color_tip: FloatVectorProperty(name="Tip", subtype='COLOR', size=3, min=0.0, soft_max=1.0, default=(1.0, 1.0, 1.0))


_COLOR_PROP_NAMES = ("color_root", "color_root_third", "color_tip_third", "color_tip")


def _description_to_object_props(obj, raw):
    values = _description_get_editable(raw)
    props = obj.engine_hair
    props.lod_distance = values["lod_distance"]
    props.lod_reduction = values["lod_reduction"]
    props.tess_max = values["tess_max"]
    props.tess_min = values["tess_min"]
    props.clump_max = values["strands_per_clump_max"]
    props.clump_min = values["strands_per_clump_min"]
    props.strand_roundness = values["strand_roundness"]
    props.strand_length_random = values["strand_length_random"]
    props.strand_thickness_random = values["strand_thickness_random"]
    props.clump_roundness = values["clump_roundness"]
    props.gloss = values["gloss"]
    props.simulation_enable = values["simulation_enable"]
    props.curls_enable = values["curls_enable"]
    props.sim_stiffness_length = values["simulation_stiffness_length"]
    props.sim_stiffness_power = values["simulation_stiffness_power"]
    props.sim_drag = values["simulation_drag"]
    for name, color in zip(_COLOR_PROP_NAMES, values["color_points"]):
        setattr(props, name, tuple(max(0.0, float(c)) for c in color))
    props.color_simple = tuple(props.color_root)


def _description_from_object_props(obj, raw):
    props = obj.engine_hair
    values = {
        "lod_distance": props.lod_distance,
        "lod_reduction": props.lod_reduction,
        "tess_max": props.tess_max,
        "tess_min": props.tess_min,
        "strands_per_clump_max": props.clump_max,
        "strands_per_clump_min": props.clump_min,
        "strand_roundness": props.strand_roundness,
        "strand_length_random": props.strand_length_random,
        "strand_thickness_random": props.strand_thickness_random,
        "clump_roundness": props.clump_roundness,
        "gloss": props.gloss,
        "simulation_enable": props.simulation_enable,
        "curls_enable": props.curls_enable,
        "simulation_stiffness_length": props.sim_stiffness_length,
        "simulation_stiffness_power": props.sim_stiffness_power,
        "simulation_drag": props.sim_drag,
        "color_points": [list(getattr(props, name)) for name in _COLOR_PROP_NAMES],
    }
    return _description_apply_editable(raw, values)




def _template_meters_per_unit(template):
    built = template.payload(BLOCK_HASHES["ModelBuilt"])
    return struct.unpack_from("<f", built, MODEL_BUILT_COMMON_MPU_OFFSET)[0]


def _read_strand_subsets(template):
    subsets_hash = BLOCK_HASHES["ModelStrandSubsets"]
    if subsets_hash not in template.blocks:
        return []
    payload = template.payload(subsets_hash)
    offset, _size = template.blocks[subsets_hash]
    count = len(payload) // SUBSET_SIZE
    result = []
    for index in range(count):
        base = index * SUBSET_SIZE
        name_hash, name_rel = struct.unpack_from("<II", payload, base)
        strand_count, strand_start = struct.unpack_from(
            "<II", payload, base + SUBSET_STRAND_COUNT_OFF
        )
        pack = struct.unpack_from("<I", payload, base + SUBSET_SKINNING_PACK_OFF)[0]
        name = ""
        if name_rel:
            end = template.data.find(b"\0", name_rel)
            if end >= 0:
                name = template.data[name_rel:end].decode("ascii", errors="replace")
        raw_description = bytes(payload[base + SUBSET_DESCRIPTION_OFF : base + SUBSET_DESCRIPTION_OFF + DESCRIPTION_SIZE])
        result.append(
            {
                "index": index,
                "name": name or f"0x{name_hash:08X}",
                "name_hash": name_hash,
                "strand_count": strand_count,
                "strand_start": strand_start,
                "skinning_type": pack & 0x3,
                "look_id": _sign_extend((pack >> 2) & 0x3FF, 10),
                "lod_id": (pack >> 12) & 0xF,
                "lod_mask": (pack >> 16) & 0x3F,
                "description": raw_description,
            }
        )
    return result


def _read_subset_strands(template, subset, joint_names, mpu):
    strand_hash = BLOCK_HASHES["ModelStrands"]
    cv_hash = BLOCK_HASHES["ModelStrandCVs"]
    jb_hash = BLOCK_HASHES["ModelStrandJBs"]
    cvw_hash = BLOCK_HASHES["ModelStrandCVWeights"]
    sb_hash = BLOCK_HASHES["ModelStrandSBs"]
    strand_off, _ = template.blocks[strand_hash]
    cv_off, _ = template.blocks[cv_hash]
    jb_off, jb_size = template.blocks.get(jb_hash, (0, 0))
    cvw_off, cvw_size = template.blocks.get(cvw_hash, (0, 0))
    sb_off, sb_size = template.blocks.get(sb_hash, (0, 0))
    has_joint_data = subset["skinning_type"] == SKINNING_JOINTS and jb_size and cvw_size
    has_skin_binding = subset["skinning_type"] == SKINNING_GEOMETRY and sb_size

    strands = []
    for local_index in range(subset["strand_count"]):
        strand_index = subset["strand_start"] + local_index
        record = _unpack_strand(template.data, strand_off + strand_index * STRAND_SIZE)
        uv = (record["uv"][0] / 65535.0, record["uv"][1] / 65535.0)
        basis_u = _unpack_normal_bytes(record["basis_u_bytes"])
        basis_v = _unpack_normal_bytes(record["basis_v_bytes"])

        positions, normals = [], []
        for cv_local in range(record["cv_count"]):
            cv_index = record["cv_start"] + cv_local
            pos_i16, nrm_bytes = _unpack_cv(template.data, cv_off + cv_index * CV_SIZE)
            positions.append(_vec_from_i16(pos_i16, mpu))
            normals.append(_unpack_normal_bytes(nrm_bytes))

        joint_names_for_strand = []
        cv_weights = []
        if has_joint_data:
            indices = _unpack_joint_binding(template.data, jb_off + strand_index * JOINT_BINDING_SIZE)
            joint_names_for_strand = [
                joint_names[index] if 0 <= index < len(joint_names) else ""
                for index in indices
            ]
            for cv_local in range(record["cv_count"]):
                cv_index = record["cv_start"] + cv_local
                weights = _unpack_cv_weights(template.data, cvw_off + cv_index * CV_WEIGHTS_SIZE)
                cv_weights.append([w if i < 5 else w * w for i, w in enumerate(weights)])

        source_subset_index = None
        if has_skin_binding:
            binding = _unpack_skin_binding(template.data, sb_off + strand_index * SKIN_BINDING_SIZE)
            source_subset_index = binding["subset_index"]

        strands.append(
            {
                "uv": uv,
                "basis_u": basis_u,
                "basis_v": basis_v,
                "positions": positions,
                "normals": normals,
                "joint_names": joint_names_for_strand,
                "cv_weights": cv_weights,
                "source_subset_index": source_subset_index,
            }
        )
    return strands


def _read_donor_joint_names(template):
    hierarchy_hash = BLOCK_HASHES["ModelJointHierarchy"]
    joint_hash = BLOCK_HASHES["ModelJoint"]
    if hierarchy_hash not in template.blocks or joint_hash not in template.blocks:
        return []
    hierarchy = template.payload(hierarchy_hash)
    joint_count = struct.unpack_from("<HHHH", hierarchy, 0)[1]
    joint_offset, joint_size = template.blocks[joint_hash]
    if joint_size < joint_count * 16:
        return []
    names = []
    for index in range(joint_count):
        record_offset = joint_offset + index * 16
        record = struct.unpack_from("<hHHHII", template.data, record_offset)
        addresses = (record_offset + int(record[5]), template.sb_offset + int(record[5]))
        name = ""
        for address in addresses:
            if not 0 <= address < len(template.data):
                continue
            raw = template.data[address:address + 256].split(b"\x00", 1)[0]
            candidate = raw.decode("ascii", errors="ignore")
            if candidate and all(character.isprintable() for character in candidate):
                name = candidate
                break
        names.append(name or f"Joint_{int(record[4]):08X}")
    return names


RIBBON_HALF_WIDTH = 0.0015
HAIR_GUIDE_MESH_KEY = "engine_hair_guide_mesh"
HAIR_GUIDE_OWNER_KEY = "engine_hair_curve_object"


def _build_weight_guide_mesh(name, strands, rotation):
    positions = []
    faces = []
    joint_weights = {}
    for strand in strands:
        strand_positions = strand["positions"]
        strand_names = strand["joint_names"]
        strand_weights = strand["cv_weights"]
        strand_u = rotation @ mathutils.Vector(strand["basis_u"])
        if strand_u.length < 1e-8:
            strand_u = mathutils.Vector((1.0, 0.0, 0.0))
        strand_u.normalize()
        prev_pair = None
        for local_index, position in enumerate(strand_positions):
            world = rotation @ mathutils.Vector(position)
            vi_left = len(positions)
            positions.append(world - strand_u * RIBBON_HALF_WIDTH)
            vi_right = len(positions)
            positions.append(world + strand_u * RIBBON_HALF_WIDTH)
            if strand_weights:
                row = strand_weights[local_index]
                for slot, joint_name in enumerate(strand_names):
                    if not joint_name or slot >= len(row) or row[slot] <= 0.0:
                        continue
                    joint_weights.setdefault(joint_name, []).append((vi_left, row[slot]))
                    joint_weights.setdefault(joint_name, []).append((vi_right, row[slot]))
            if prev_pair is not None:
                faces.append((prev_pair[0], prev_pair[1], vi_right, vi_left))
            prev_pair = (vi_left, vi_right)

    mesh = bpy.data.meshes.new(f"{name}_weights")
    mesh.from_pydata([tuple(p) for p in positions], [], faces)
    mesh.update(calc_edges=True)
    return mesh, joint_weights


def _build_weight_guide_mesh_from_curve(hair_obj):
    curve_data = hair_obj.data
    positions_attr = curve_data.attributes["position"].data
    basis_u_attr = curve_data.attributes.get("hair_basis_u")
    joint_attrs = [curve_data.attributes.get(f"hair_joint_{slot}") for slot in range(MAX_STRAND_JOINT_WEIGHTS)]
    weight_attrs = [curve_data.attributes.get(f"hair_weight_{slot}") for slot in range(MAX_STRAND_JOINT_WEIGHTS)]

    positions = []
    faces = []
    joint_weights = {}
    for curve_index in range(len(curve_data.curves)):
        start, end = _curve_point_range(curve_data, curve_index)
        strand_u = mathutils.Vector(basis_u_attr.data[curve_index].vector) if basis_u_attr is not None else mathutils.Vector((1.0, 0.0, 0.0))
        if strand_u.length < 1e-8:
            strand_u = mathutils.Vector((1.0, 0.0, 0.0))
        strand_u.normalize()

        names = []
        for slot in range(MAX_STRAND_JOINT_WEIGHTS):
            attr = joint_attrs[slot]
            name = attr.data[curve_index].value if attr is not None else ""
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            names.append(name)

        prev_pair = None
        for point_index in range(start, end):
            world = mathutils.Vector(positions_attr[point_index].vector)
            vi_left = len(positions)
            positions.append(world - strand_u * RIBBON_HALF_WIDTH)
            vi_right = len(positions)
            positions.append(world + strand_u * RIBBON_HALF_WIDTH)
            for slot, joint_name in enumerate(names):
                if not joint_name:
                    continue
                weight = weight_attrs[slot].data[point_index].value if weight_attrs[slot] is not None else 0.0
                if weight <= 0.0:
                    continue
                joint_weights.setdefault(joint_name, []).append((vi_left, weight))
                joint_weights.setdefault(joint_name, []).append((vi_right, weight))
            if prev_pair is not None:
                faces.append((prev_pair[0], prev_pair[1], vi_right, vi_left))
            prev_pair = (vi_left, vi_right)

    mesh = bpy.data.meshes.new(f"{hair_obj.name}_weights")
    mesh.from_pydata([tuple(p) for p in positions], [], faces)
    mesh.update(calc_edges=True)
    return mesh, joint_weights


def _attach_guide_mesh(hair_obj, arm, context, mesh, joint_weights):
    guide_obj = bpy.data.objects.new(f"{hair_obj.name}_weights", mesh)
    guide_obj.parent = hair_obj
    for joint_name, pairs in joint_weights.items():
        group = guide_obj.vertex_groups.new(name=joint_name)
        totals = {}
        for vertex_index, weight in pairs:
            totals[vertex_index] = totals.get(vertex_index, 0.0) + weight
        for vertex_index, weight in totals.items():
            group.add([vertex_index], min(1.0, weight), 'REPLACE')
    guide_obj[HAIR_GUIDE_OWNER_KEY] = hair_obj.name
    hair_obj[HAIR_GUIDE_MESH_KEY] = guide_obj.name
    context.scene.collection.objects.link(guide_obj)
    armature_modifier = guide_obj.modifiers.new("Armature", 'ARMATURE')
    armature_modifier.object = arm
    guide_obj.display_type = 'WIRE'
    guide_obj.hide_render = True
    return guide_obj


def _read_guide_mesh_weights(guide_obj):
    mesh = guide_obj.data
    group_names = [group.name for group in guide_obj.vertex_groups]
    cv_count = len(mesh.vertices) // 2
    per_cv = []
    for cv_index in range(cv_count):
        combined = {}
        for vertex_index in (cv_index * 2, cv_index * 2 + 1):
            for assignment in mesh.vertices[vertex_index].groups:
                if assignment.group < len(group_names):
                    name = group_names[assignment.group]
                    combined[name] = combined.get(name, 0.0) + assignment.weight * 0.5
        per_cv.append(combined)
    return per_cv


# Dunno if i should keep this since it was useful for debug


def import_strand_hair(filepath, arm, context):
    if arm is None or arm.type != "ARMATURE":
        raise ValueError("Please select the character armature first.")
    template = _Dat1Template(filepath)
    subsets = _read_strand_subsets(template)
    if not subsets:
        raise ValueError(f"{os.path.basename(filepath)} does not contain any strand hair data.")

    joint_names = _read_donor_joint_names(template)
    mpu = _template_meters_per_unit(template)
    rotation = SWIZZLE_MAT.to_3x3()

    created = []
    for subset in subsets:
        if subset["skinning_type"] not in (SKINNING_JOINTS, SKINNING_GEOMETRY):
            continue
        if subset["lod_id"] != 0:
            continue  # only the highest-detail LOD; every export writes back as LOD0-only anyway
        strands = _read_subset_strands(template, subset, joint_names, mpu)
        if not strands:
            continue

        curve_data = bpy.data.hair_curves.new(subset["name"])
        curve_data.add_curves([len(strand["positions"]) for strand in strands])

        position_flat = []
        normal_flat = []
        for strand in strands:
            for position in strand["positions"]:
                world = rotation @ mathutils.Vector(position)
                position_flat.extend((world.x, world.y, world.z))
            for normal in strand["normals"]:
                world = rotation @ mathutils.Vector(normal)
                normal_flat.extend((world.x, world.y, world.z))
        curve_data.attributes["position"].data.foreach_set("vector", position_flat)

        normal_attr = curve_data.attributes.new("hair_normal", "FLOAT_VECTOR", "POINT")
        normal_attr.data.foreach_set("vector", normal_flat)

        weight_attrs = [
            curve_data.attributes.new(f"hair_weight_{slot}", "FLOAT", "POINT")
            for slot in range(MAX_STRAND_JOINT_WEIGHTS)
        ]
        for slot in range(MAX_STRAND_JOINT_WEIGHTS):
            flat = [
                (strand["cv_weights"][cv_index][slot] if strand["cv_weights"] else 0.0)
                for strand in strands
                for cv_index in range(len(strand["positions"]))
            ]
            weight_attrs[slot].data.foreach_set("value", flat)

        uv_attr = curve_data.attributes.new("hair_uv", "FLOAT2", "CURVE")
        uv_attr.data.foreach_set("vector", [component for strand in strands for component in strand["uv"]])

        basis_u_attr = curve_data.attributes.new("hair_basis_u", "FLOAT_VECTOR", "CURVE")
        basis_v_attr = curve_data.attributes.new("hair_basis_v", "FLOAT_VECTOR", "CURVE")
        basis_u_flat, basis_v_flat = [], []
        for strand in strands:
            world_u = rotation @ mathutils.Vector(strand["basis_u"])
            world_v = rotation @ mathutils.Vector(strand["basis_v"])
            basis_u_flat.extend((world_u.x, world_u.y, world_u.z))
            basis_v_flat.extend((world_v.x, world_v.y, world_v.z))
        basis_u_attr.data.foreach_set("vector", basis_u_flat)
        basis_v_attr.data.foreach_set("vector", basis_v_flat)

        joint_attrs = [
            curve_data.attributes.new(f"hair_joint_{slot}", "STRING", "CURVE")
            for slot in range(MAX_STRAND_JOINT_WEIGHTS)
        ]
        for slot in range(MAX_STRAND_JOINT_WEIGHTS):
            for strand_index, strand in enumerate(strands):
                names = strand["joint_names"]
                name = names[slot] if slot < len(names) else ""
                joint_attrs[slot].data[strand_index].value = name.encode("utf-8")

        obj = bpy.data.objects.new(subset["name"], curve_data)
        obj.parent = arm
        obj[HAIR_SUBSET_KEY] = subset["name"]
        obj[HAIR_SKINNING_KEY] = SKINNING_NAMES.get(subset["skinning_type"], "joints")
        obj[HAIR_SOURCE_KEY] = os.path.basename(filepath)
        obj[HAIR_DESCRIPTION_KEY] = base64.b64encode(subset["description"]).decode("ascii")
        _description_to_object_props(obj, subset["description"])
        context.scene.collection.objects.link(obj)

        if subset["skinning_type"] == SKINNING_JOINTS:
            guide_mesh, joint_weights = _build_weight_guide_mesh(subset["name"], strands, rotation)
            _attach_guide_mesh(obj, arm, context, guide_mesh, joint_weights)

        if subset["skinning_type"] == SKINNING_GEOMETRY:
            scalp_candidate = _find_matching_scalp_object(arm, strands)
            if scalp_candidate is not None:
                obj.engine_hair_scalp_object = scalp_candidate

        created.append(obj)

    if not created:
        raise ValueError(
            f"No compatible hair parts could be imported from {os.path.basename(filepath)}."
        )
    return created




def hair_objects_for_armature(arm):
    return [
        obj for obj in bpy.data.objects
        if obj.type == "CURVES" and obj.parent == arm and HAIR_SUBSET_KEY in obj
    ]


def _curve_point_range(curve_data, curve_index):
    curve = curve_data.curves[curve_index]
    start = curve.first_point_index
    return start, start + curve.points_length


def _find_matching_scalp_object(arm, strands):
    votes = {}
    for strand in strands:
        index = strand.get("source_subset_index")
        if index is not None:
            votes[index] = votes.get(index, 0) + 1
    if not votes:
        return None
    winner = max(votes, key=votes.get)
    for candidate in arm.children_recursive:
        if candidate.type == "MESH" and candidate.get("engine_subset_index") == winner:
            return candidate
    return None


def _scalp_bvh(scalp_obj):
    mesh = scalp_obj.data
    mesh.calc_loop_triangles()
    matrix = scalp_obj.matrix_world
    positions = [matrix @ vertex.co for vertex in mesh.vertices]
    triangles = [tuple(int(v) for v in triangle.vertices) for triangle in mesh.loop_triangles]
    if not triangles:
        raise ValueError(f"{scalp_obj.name} has no faces for attaching hair roots.")
    bvh = mathutils.bvhtree.BVHTree.FromPolygons(positions, triangles, all_triangles=True)
    return bvh, triangles, positions


def _barycentric_weights(point, t0, t1, t2):
    normal = (t1 - t0).cross(t2 - t0)
    length = normal.length
    if length < 1e-12:
        return (1.0, 0.0, 0.0)
    normal = normal / length
    d0 = normal.dot((t1 - point).cross(t2 - point))
    d1 = normal.dot((t2 - point).cross(t0 - point))
    d2 = normal.dot((t0 - point).cross(t1 - point))
    total = d0 + d1 + d2
    if abs(total) < 1e-12:
        return (1.0, 0.0, 0.0)
    return (d0 / total, d1 / total, d2 / total)


def rebind_geometry_hair(hair_obj, scalp_obj):
    bvh, triangles, _positions = _scalp_bvh(scalp_obj)
    curve_data = hair_obj.data
    positions = curve_data.attributes["position"].data
    subset_index = int(scalp_obj.get("engine_subset_index", 0))
    bindings = []
    unbound = 0
    for curve_index in range(len(curve_data.curves)):
        start, _end = _curve_point_range(curve_data, curve_index)
        root_local = mathutils.Vector(positions[start].vector)
        root_world = hair_obj.matrix_world @ root_local
        location, _normal, tri_index, _distance = bvh.find_nearest(root_world)
        if tri_index is None:
            unbound += 1
            bindings.append((0, 0, 0, 8, 8, subset_index))
            continue
        v0, v1, v2 = triangles[tri_index]
        t0 = scalp_obj.matrix_world @ scalp_obj.data.vertices[v0].co
        t1 = scalp_obj.matrix_world @ scalp_obj.data.vertices[v1].co
        t2 = scalp_obj.matrix_world @ scalp_obj.data.vertices[v2].co
        w0, w1, _w2 = _barycentric_weights(location, t0, t1, t2)
        weight0 = max(0, min(15, int(max(0.0, min(1.0, w0)) * 15.0 + 0.5)))
        weight1 = max(0, min(15, int(max(0.0, min(1.0, w1)) * 15.0 + 0.5)))
        bindings.append((v0, v1, v2, weight0, weight1, subset_index))
    return bindings, unbound


def compile_export_hair(arm):
    objects = hair_objects_for_armature(arm)
    if not objects:
        return None, []

    joint_by_name = {bone.name: bone.get("engine_joint_index", -1) for bone in arm.data.bones}
    rotation_inv = SWIZZLE_MAT.to_3x3().inverted()

    mpu_source = float(arm.get("engine_model_source_common_mpu", 0.0) or arm.get("engine_mpu", 0.0) or 0.0)
    if mpu_source <= 0.0:
        raise ValueError("Character scaling information is missing. Please re-import the model.")
    inv_mpu = 1.0 / mpu_source

    subset_bytes = bytearray()
    strand_bytes = bytearray()
    cv_bytes = bytearray()
    jb_bytes = bytearray()
    cvw_bytes = bytearray()
    sb_bytes = bytearray()
    strand_cursor = 0
    cv_cursor = 0
    warnings = []
    any_geometry = False

    for obj in objects:
        curve_data = obj.data
        subset_name = str(obj.get(HAIR_SUBSET_KEY) or obj.name)
        skinning_type = SKINNING_IDS.get(str(obj.get(HAIR_SKINNING_KEY) or "joints"), SKINNING_JOINTS)

        scalp_obj = None
        root_bindings = None
        if skinning_type == SKINNING_GEOMETRY:
            scalp_obj = getattr(obj, "engine_hair_scalp_object", None)
            if scalp_obj is None or scalp_obj.type != 'MESH':
                raise ValueError(
                    f"Please select a Scalp Mesh for {obj.name} in the Strand Hair panel before exporting."
                )
            root_bindings, unbound = rebind_geometry_hair(obj, scalp_obj)
            if unbound:
                warnings.append(f"{obj.name} has {unbound} hair strand(s) that could not reach the scalp surface.")
            any_geometry = True

        positions = curve_data.attributes["position"].data
        normal_attr = curve_data.attributes.get("hair_normal")
        weight_attrs = [curve_data.attributes.get(f"hair_weight_{slot}") for slot in range(MAX_STRAND_JOINT_WEIGHTS)]
        uv_attr = curve_data.attributes.get("hair_uv")
        basis_u_attr = curve_data.attributes.get("hair_basis_u")
        basis_v_attr = curve_data.attributes.get("hair_basis_v")
        joint_attrs = [curve_data.attributes.get(f"hair_joint_{slot}") for slot in range(MAX_STRAND_JOINT_WEIGHTS)]


        guide_weights_per_cv = None
        guide_name = obj.get(HAIR_GUIDE_MESH_KEY)
        if guide_name:
            guide_obj = bpy.data.objects.get(guide_name)
            if guide_obj is not None and guide_obj.type == 'MESH':
                guide_weights_per_cv = _read_guide_mesh_weights(guide_obj)
                if len(guide_weights_per_cv) != len(curve_data.points):
                    raise ValueError(
                        f"Guide mesh {guide_obj.name} vertex count does not match the hair points on {obj.name}. "
                        "Please paint weights without adding or deleting vertices."
                    )

        curve_count = len(curve_data.curves)
        for curve_index in range(curve_count):
            start, end = _curve_point_range(curve_data, curve_index)
            cv_count = end - start
            if cv_count <= 0 or cv_count > 255:
                raise ValueError(f"{obj.name} curve {curve_index} has {cv_count} points, but strands should have between 1 and 255 points.")

            uv_raw = (0.0, 0.0)
            if uv_attr is not None:
                uv_raw = tuple(uv_attr.data[curve_index].vector)
            basis_u = (0.0, 0.0, 0.0) if basis_u_attr is None else tuple(basis_u_attr.data[curve_index].vector)
            basis_v = (0.0, 0.0, 0.0) if basis_v_attr is None else tuple(basis_v_attr.data[curve_index].vector)
            basis_u = tuple(rotation_inv @ mathutils.Vector(basis_u))
            basis_v = tuple(rotation_inv @ mathutils.Vector(basis_v))

            strand_bytes += _pack_strand(
                cv_cursor,
                cv_count,
                (uv_raw[0] * 65535.0, uv_raw[1] * 65535.0),
                _pack_normal_bytes(basis_u),
                _pack_normal_bytes(basis_v),
            )

            if skinning_type == SKINNING_JOINTS:
                if guide_weights_per_cv is not None:
                    cv_rows = guide_weights_per_cv[start:end]
                    active_names = sorted({name for row in cv_rows for name in row})
                    missing = [name for name in active_names if name not in joint_by_name]
                    if missing:
                        raise ValueError(f"{obj.name} uses bones that are not in the character skeleton ({', '.join(missing)}).")
                    raw_weights = [[row.get(name, 0.0) for name in active_names] for row in cv_rows]
                else:
                    names = []
                    for slot in range(MAX_STRAND_JOINT_WEIGHTS):
                        attr = joint_attrs[slot]
                        name = attr.data[curve_index].value if attr is not None else ""
                        if isinstance(name, bytes):
                            name = name.decode("utf-8", errors="ignore")
                        names.append(name)
                    active_names = [name for name in names if name]
                    missing = [name for name in active_names if name not in joint_by_name]
                    if missing:
                        raise ValueError(f"{obj.name} uses bones that are not in the character skeleton ({', '.join(missing)}).")

                    raw_weights = []
                    for cv_local in range(cv_count):
                        point_index = start + cv_local
                        row = [
                            (weight_attrs[slot].data[point_index].value if weight_attrs[slot] is not None else 0.0)
                            for slot in range(MAX_STRAND_JOINT_WEIGHTS)
                        ][: len(active_names)] if active_names else []
                        raw_weights.append(row)

                if active_names and raw_weights and any(any(row) for row in raw_weights):
                    quantized, kept_order = _quantize_strand_weights(raw_weights)
                    kept_names = [active_names[index] for index in kept_order]
                    jb_bytes += _pack_joint_binding([joint_by_name[name] for name in kept_names])
                    for row in quantized:
                        cvw_bytes += _pack_cv_weights(row)
                else:
                    jb_bytes += _pack_joint_binding([])
                    for _ in range(cv_count):
                        cvw_bytes += _pack_cv_weights([0.0] * MAX_STRAND_JOINT_WEIGHTS)
            else:
                jb_bytes += _pack_joint_binding([])
                for _ in range(cv_count):
                    cvw_bytes += _pack_cv_weights([0.0] * MAX_STRAND_JOINT_WEIGHTS)

            if skinning_type == SKINNING_GEOMETRY:
                v0, v1, v2, weight0, weight1, subset_index = root_bindings[curve_index]
                sb_bytes += _pack_skin_binding(v0, v1, v2, weight0, weight1, subset_index)
            else:
                sb_bytes += _pack_skin_binding(0, 0, 0, 0, 0, 0)

            for cv_local in range(cv_count):
                point_index = start + cv_local
                position = mathutils.Vector(positions[point_index].vector)
                position = rotation_inv @ position
                normal = (0.0, 1.0, 0.0)
                if normal_attr is not None:
                    normal = tuple(rotation_inv @ mathutils.Vector(normal_attr.data[point_index].vector))
                cv_bytes += _pack_cv(_vec_to_i16(tuple(position), inv_mpu), _pack_normal_bytes(normal))
            cv_cursor += cv_count

        raw_description = b""
        encoded = obj.get(HAIR_DESCRIPTION_KEY)
        if encoded:
            try:
                raw_description = base64.b64decode(encoded)
            except Exception:
                raw_description = b""
        description = _description_from_object_props(obj, raw_description)

        subset_record = bytearray(SUBSET_SIZE)
        struct.pack_into("<I", subset_record, SUBSET_NAME_HASH_OFF, string_crc32(subset_name))
        struct.pack_into("<I", subset_record, SUBSET_STRAND_COUNT_OFF, curve_count)
        struct.pack_into("<I", subset_record, SUBSET_STRAND_START_OFF, strand_cursor)
        pack = (skinning_type & 0x3) | (0x3FF << 2) | (0 << 12) | (0b1 << 16)
        struct.pack_into("<I", subset_record, SUBSET_SKINNING_PACK_OFF, pack)
        subset_record[SUBSET_DESCRIPTION_OFF : SUBSET_DESCRIPTION_OFF + DESCRIPTION_SIZE] = description
        subset_bytes += subset_record
        strand_cursor += curve_count

    if not subset_bytes:
        return None, warnings

    blocks = {
        BLOCK_HASHES["ModelStrandSubsets"]: bytes(subset_bytes),
        BLOCK_HASHES["ModelStrands"]: bytes(strand_bytes),
        BLOCK_HASHES["ModelStrandCVs"]: bytes(cv_bytes),
        BLOCK_HASHES["ModelStrandJBs"]: bytes(jb_bytes),
        BLOCK_HASHES["ModelStrandCVWeights"]: bytes(cvw_bytes),
    }
    if any_geometry:
        blocks[BLOCK_HASHES["ModelStrandSBs"]] = bytes(sb_bytes)
    return blocks, warnings


# ---------------------------------------------------------------------------
# operators + panel
# ---------------------------------------------------------------------------


class MODEL_OT_import_strand_hair(Operator, ImportHelper):
    bl_idname = "model.import_strand_hair"
    bl_label = "Import Strand Hair From Model"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})

    def execute(self, context):
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select the character armature first.")
            return {'CANCELLED'}
        try:
            created = import_strand_hair(self.filepath, arm, context)
        except Exception as exc:
            log_exception("Strand hair import failed")
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        total_strands = sum(len(obj.data.curves) for obj in created)
        self.report(
            {'INFO'},
            f"Imported {len(created)} hair part(s) with {total_strands} total strands.",
        )
        return {'FINISHED'}


class MODEL_OT_apply_hair_simple_color(Operator):
    bl_idname = "model.apply_hair_simple_color"
    bl_label = "Apply Color To All 4 Points"
    bl_description = "Apply one solid color across the full root to tip gradient"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'CURVES':
            return {'CANCELLED'}
        props = obj.engine_hair
        color = tuple(props.color_simple)
        for name in _COLOR_PROP_NAMES:
            setattr(props, name, color)
        self.report({'INFO'}, "Updated hair color across all gradient points.")
        return {'FINISHED'}


class MODEL_OT_create_hair_guide_mesh(Operator):
    bl_idname = "model.create_hair_guide_mesh"
    bl_label = "Create Weight Guide Mesh"
    bl_description = (
        "Build a weight paintable guide mesh for this hair from its existing bone weight data"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        arm = obj.parent if obj and obj.type == 'CURVES' else None
        if obj is None or obj.type != 'CURVES' or arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select a hair object that is parented to the armature.")
            return {'CANCELLED'}
        existing_name = obj.get(HAIR_GUIDE_MESH_KEY)
        if existing_name and bpy.data.objects.get(existing_name) is not None:
            self.report({'INFO'}, f"{obj.name} already has a guide mesh ({existing_name}).")
            return {'CANCELLED'}
        mesh, joint_weights = _build_weight_guide_mesh_from_curve(obj)
        guide_obj = _attach_guide_mesh(obj, arm, context, mesh, joint_weights)
        self.report({'INFO'}, f"Created {guide_obj.name} with {len(joint_weights)} bone groups.")
        return {'FINISHED'}


class MODEL_OT_select_hair_guide_mesh(Operator):
    bl_idname = "model.select_hair_guide_mesh"
    bl_label = "Weight Paint This Hair"
    bl_description = "Select this hair's guide mesh and enter Weight Paint mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        guide_name = obj.get(HAIR_GUIDE_MESH_KEY) if obj else None
        guide_obj = bpy.data.objects.get(guide_name) if guide_name else None
        if guide_obj is None:
            self.report({'ERROR'}, "Could not find a guide mesh to select.")
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for other in context.selected_objects:
            other.select_set(False)
        guide_obj.hide_set(False)
        guide_obj.select_set(True)
        context.view_layer.objects.active = guide_obj
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        return {'FINISHED'}


def _poll_scalp_object(self, obj):
    return obj.type == 'MESH'


def register_hair_properties():
    bpy.types.Object.engine_hair_scalp_object = PointerProperty(
        name="Scalp Mesh",
        description=(
            "The imported mesh part this geometry skinned hair attaches to. "
            "Must be an unedited mesh part parented to the same character armature."
        ),
        type=bpy.types.Object,
        poll=_poll_scalp_object,
    )
    bpy.types.Object.engine_hair = PointerProperty(type=MODEL_PG_hair_description)


def unregister_hair_properties():
    for name in ("engine_hair_scalp_object", "engine_hair"):
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)


class ModelHairPanel(Panel):
    bl_label = "Strand Hair"
    bl_idname = "OBJECT_PT_luna_engine_hair"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Luna Engine'

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        return bool(obj and obj.type in ('ARMATURE', 'CURVES'))

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        arm = obj if obj.type == 'ARMATURE' else obj.parent
        if obj.type == 'CURVES':
            layout.label(text=f"Subset: {obj.get(HAIR_SUBSET_KEY, obj.name)}")
            layout.label(text=f"Skinning: {obj.get(HAIR_SKINNING_KEY, '?')}")
            layout.label(text=f"Strands: {len(obj.data.curves)}")
            layout.label(text=f"Source: {obj.get(HAIR_SOURCE_KEY, '?')}")

            if obj.get(HAIR_SKINNING_KEY) == "joints":
                guide_name = obj.get(HAIR_GUIDE_MESH_KEY)
                guide_obj = bpy.data.objects.get(guide_name) if guide_name else None
                box = layout.box()
                if guide_obj is not None:
                    box.label(text=guide_obj.name, icon='MESH_DATA')
                    box.operator(MODEL_OT_select_hair_guide_mesh.bl_idname, icon='WPAINT_HLT')
                    box.label(text="Vertex groups match the character bones.")
                else:
                    box.label(text="No weight paint guide mesh found.", icon='INFO')
                    box.operator(MODEL_OT_create_hair_guide_mesh.bl_idname, icon='ADD')

            if obj.get(HAIR_SKINNING_KEY) == "geometry":
                box = layout.box()
                box.label(text="Root Binding", icon='STICKY_UVS_LOC')
                box.prop(obj, "engine_hair_scalp_object", text="Scalp Mesh")
                if obj.engine_hair_scalp_object is not None:
                    box.label(
                        text=f"Rebinds against {obj.engine_hair_scalp_object.name} on export",
                        icon='INFO',
                    )
                    box.label(
                        text="Hair roots follow the scalp pose in the game engine.",
                        icon='INFO',
                    )

            if HAIR_DESCRIPTION_KEY in obj:
                props = obj.engine_hair
                box = layout.box()
                box.label(text="Hair Description", icon='PARTICLES')
                box.prop(props, "lod_distance")
                box.prop(props, "lod_reduction")
                box.prop(props, "tess_min")
                box.prop(props, "tess_max")
                box.prop(props, "clump_min")
                box.prop(props, "clump_max")
                box.prop(props, "strand_roundness")
                box.prop(props, "strand_length_random")
                box.prop(props, "strand_thickness_random")
                box.prop(props, "clump_roundness")
                box.prop(props, "gloss")
                box.prop(props, "simulation_enable")
                box.prop(props, "curls_enable")
                if props.simulation_enable:
                    box.prop(props, "sim_stiffness_length")
                    box.prop(props, "sim_stiffness_power")
                    box.prop(props, "sim_drag")

                color_box = layout.box()
                color_box.label(text="Color", icon='COLOR')
                color_box.prop(props, "color_simple", text="")
                color_box.label(
                    text="Applying will set all 4 gradient points to this color.", icon='INFO'
                )
                color_box.operator(MODEL_OT_apply_hair_simple_color.bl_idname)
                advanced = color_box.box()
                advanced.label(text="Advanced: Root To Tip Gradient")
                for name in _COLOR_PROP_NAMES:
                    advanced.prop(props, name)
            return
        if arm is None or arm.type != 'ARMATURE':
            layout.label(text="Please select the character armature.", icon='INFO')
            return
        layout.operator(MODEL_OT_import_strand_hair.bl_idname, icon='IMPORT')
        subsets = hair_objects_for_armature(arm)
        if subsets:
            box = layout.box()
            for subset_obj in subsets:
                box.label(text=f"{subset_obj.name}: {len(subset_obj.data.curves)} strands")
