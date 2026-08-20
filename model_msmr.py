"""Marvel's Spider-Man Remastered model stream decoding.

MSMR predates the per-subset geometry container used by Spider-Man 2.  Its
subsets address shared, delta-compressed vertex and index streams instead.
This module deliberately has no Blender dependency so the binary decoder can
be checked against extracted game assets on its own.
"""

import math
import struct

import numpy as np


MSMR_MODEL_MAGIC = 0x98906B9F
MSMR_SUBSET_RECORD_SIZE = 64
MSMR_VERTEX_STREAM_COUNT = 8
MSMR_VERTEX_STREAM_MAX_SIZE = 0x10000
MSMR_VERTEX_CHUNK_SIZE = MSMR_VERTEX_STREAM_COUNT * MSMR_VERTEX_STREAM_MAX_SIZE
MSMR_SKIN_BATCH_RECORD_SIZE = 16
MSMR_SKIN_BATCH_VERTEX_COUNT = 16
MSMR_JOINT_LOOKUP_RECORD_SIZE = 8
MSMR_MIRROR_ID_RECORD_SIZE = 4

# Engine CRC32 tags.  The names are also registered in hashes.BLOCK_HASHES;
# keeping the numeric values here lets this decoder remain Blender-independent.
MSMR_MODEL_BUILT_HASH = 0x283D0383
MSMR_MODEL_LOOK_HASH = 0x06EB7EFC
MSMR_MODEL_LOOK_BUILT_HASH = 0x811902D7
MSMR_MODEL_LOOK_GROUP_HASH = 0x4CCEA4AD
MSMR_MODEL_SUBSET_HASH = 0x78D9CBDE
MSMR_MODEL_INDEX_HASH = 0x0859863D
MSMR_MODEL_STD_VERT_HASH = 0xA98BE69B
MSMR_MODEL_UV1_VERT_HASH = 0x6B855EED
MSMR_MODEL_COL_VERT_HASH = 0x5CBA9DE9
MSMR_MODEL_SKIN_BATCH_HASH = 0xC61B1FF5
MSMR_MODEL_SKIN_DATA_HASH = 0xDCA379A2
MSMR_MODEL_SKIN_JOINT_REMAP_HASH = 0x5240C82B
MSMR_MODEL_JOINT_LOOKUP_HASH = 0xEE31971C
MSMR_MODEL_MIRROR_IDS_HASH = 0xC5354B60
MSMR_MODEL_LOCATOR_HASH = 0x9F614FAB
MSMR_MODEL_LOCATOR_LOOKUP_HASH = 0x731CBC2E
MSMR_MODEL_SPLINE_SUBSETS_HASH = 0x3C9DABDF
MSMR_MODEL_SPLINES_HASH = 0x27CA5246
MSMR_MODEL_SPLINE_POINTS_HASH = 0xB25B3163
MSMR_MODEL_SPLINE_SKIN_BINDING_HASH = 0xBB7303D5
MSMR_MODEL_SPLINE_JOINT_BINDING_HASH = 0x14D8B13C
MSMR_MODEL_SPLINE_JOINT_WEIGHTS_HASH = 0x5D5CF541
MSMR_MODEL_ANIM_MORPH_INFO_HASH = 0x380A5744
MSMR_MODEL_ANIM_MORPH_DATA_HASH = 0x5E709570
MSMR_MODEL_ANIM_MORPH_INDICES_HASH = 0xA600C108
MSMR_MORPH_PAGE_VERTEX_COUNT = 0xA00
MSMR_LOCATOR_RECORD_SIZE = 64
MSMR_LOCATOR_LOOKUP_RECORD_SIZE = 8
MSMR_LOOK_GROUP_RECORD_SIZE = 24
MSMR_SPLINE_SUBSET_RECORD_SIZE = 0x4E8
MSMR_SPLINE_RECORD_SIZE = 12
MSMR_SPLINE_POINT_RECORD_SIZE = 8
MSMR_SPLINE_SKIN_BINDING_RECORD_SIZE = 8


def is_msmr_model(blocks):
    """Return whether a DAT1 block table uses the Remastered model layout."""
    return (
        MSMR_MODEL_SUBSET_HASH in blocks
        and MSMR_MODEL_INDEX_HASH in blocks
        and MSMR_MODEL_STD_VERT_HASH in blocks
    )


def _checked_block(data, blocks, block_hash, label):
    block = blocks.get(block_hash)
    if not block:
        raise ValueError(f"MSMR model is missing {label}")
    offset, size = (int(block[0]), int(block[1]))
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError(f"MSMR {label} block is outside the DAT1 payload")
    return offset, size


def _decode_msmr_directions(encoded_x, encoded_y, positive_z):
    x = np.asarray(encoded_x, dtype=np.float32)
    y = np.asarray(encoded_y, dtype=np.float32)
    x = x * np.float32(0.00276483595) - np.float32(math.sqrt(2.0))
    y = y * np.float32(0.00276483595) - np.float32(math.sqrt(2.0))
    xy_sq = x * x + y * y
    scale = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - np.float32(0.25) * xy_sq))
    z = np.float32(1.0) - np.float32(0.5) * xy_sq
    z = np.where(positive_z, z, -z)
    return np.column_stack((x * scale, y * scale, z)).astype(np.float32, copy=False)


def _decode_msmr_normals(words):
    words = np.asarray(words, dtype=np.uint32)
    return _decode_msmr_directions(
        words & np.uint32(0x3FF),
        (words >> np.uint32(10)) & np.uint32(0x3FF),
        (words & np.uint32(0x80000000)) != 0,
    )


def _decode_msmr_tangent_frame(words, position_ws, normals):
    """Decode tangent direction and bitangent handedness shared across two streams."""
    words = np.asarray(words, dtype=np.uint32)
    position_ws = np.asarray(position_ws, dtype=np.int32)
    tangent_y = np.abs(position_ws.astype(np.int64)) & 0x3FF
    tangents = _decode_msmr_directions(
        (words >> np.uint32(20)) & np.uint32(0x3FF),
        tangent_y,
        (words & np.uint32(0x40000000)) != 0,
    )

    # The encoder negates the second packed short for a positive
    # dot(cross(tangent, bitangent), normal), so its sign is inverted here.
    bitangent_signs = np.where(position_ws < 0, 1.0, -1.0).astype(np.float32)
    bitangents = np.cross(normals, tangents) * bitangent_signs[:, None]
    bitangent_lengths = np.linalg.norm(bitangents, axis=1)
    nonzero = bitangent_lengths > np.float32(1e-20)
    bitangents[nonzero] /= bitangent_lengths[nonzero, None]

    tangent_valid = ~(
        (words == np.uint32(0x7FF80200))
        & (position_ws == 0x7E)
    )
    tangents[~tangent_valid] = 0.0
    bitangents[~tangent_valid] = 0.0
    bitangent_signs[~tangent_valid] = 0.0
    return tangents, bitangents.astype(np.float32, copy=False), bitangent_signs, tangent_valid


def _decode_msmr_vertex_stream(data, offset, size):
    if size <= 0 or size % (MSMR_VERTEX_STREAM_COUNT * 2):
        raise ValueError("MSMR Model Std Vert size is not a valid eight-stream payload")

    normal_chunks = []
    value_chunks = []
    for chunk_offset in range(0, size, MSMR_VERTEX_CHUNK_SIZE):
        chunk_size = min(MSMR_VERTEX_CHUNK_SIZE, size - chunk_offset)
        if chunk_size % MSMR_VERTEX_STREAM_COUNT:
            raise ValueError("MSMR Model Std Vert has a truncated stream chunk")
        stream_size = chunk_size // MSMR_VERTEX_STREAM_COUNT
        if stream_size % 2:
            raise ValueError("MSMR Model Std Vert stream is not 16-bit aligned")
        vertex_count = stream_size // 2
        raw = memoryview(data)[offset + chunk_offset:offset + chunk_offset + chunk_size]

        normal_deltas = np.frombuffer(raw[:stream_size * 2], dtype="<u4", count=vertex_count)
        normal_chunks.append(np.bitwise_xor.accumulate(normal_deltas))

        value_deltas = np.frombuffer(
            raw[stream_size * 2:],
            dtype="<i2",
            count=vertex_count * 6,
        ).reshape(6, vertex_count)
        value_chunks.append(np.bitwise_xor.accumulate(value_deltas, axis=1))

    normal_words = np.concatenate(normal_chunks)
    values = np.concatenate(value_chunks, axis=1)
    return normal_words, values


def _msmr_position_quantization(data, blocks):
    position_scale = 1.0 / 4096.0
    position_offset = (0.0, 0.0, 0.0)
    built = blocks.get(MSMR_MODEL_BUILT_HASH)
    if built:
        built_offset, built_size = int(built[0]), int(built[1])
        if built_offset >= 0 and built_size >= 48 and built_offset + built_size <= len(data):
            candidate_offset = struct.unpack_from("<3f", data, built_offset + 28)
            candidate_scale = struct.unpack_from("<f", data, built_offset + 44)[0]
            if all(math.isfinite(value) for value in candidate_offset):
                position_offset = candidate_offset
            if math.isfinite(candidate_scale) and candidate_scale > 0.0:
                position_scale = candidate_scale
    return float(position_scale), tuple(float(value) for value in position_offset)


def decode_msmr_geometry(data, blocks):
    """Decode the shared MSMR streams into engine-space NumPy arrays."""
    index_offset, index_size = _checked_block(data, blocks, MSMR_MODEL_INDEX_HASH, "Model Index")
    vertex_offset, vertex_size = _checked_block(data, blocks, MSMR_MODEL_STD_VERT_HASH, "Model Std Vert")
    if index_size % 2:
        raise ValueError("MSMR Model Index size is not 16-bit aligned")

    index_deltas = np.frombuffer(data, dtype="<i2", count=index_size // 2, offset=index_offset).astype(np.int64)
    indices = np.bitwise_and(np.cumsum(index_deltas, dtype=np.int64), 0xFFFF).astype(np.uint16)

    normal_words, values = _decode_msmr_vertex_stream(data, vertex_offset, vertex_size)
    vertex_count = int(values.shape[1])

    position_scale, position_offset = _msmr_position_quantization(data, blocks)
    uv_scale = 1.0 / 16384.0
    built = blocks.get(MSMR_MODEL_BUILT_HASH)
    if built:
        built_offset, built_size = int(built[0]), int(built[1])
        if built_offset >= 0 and built_size >= 52 and built_offset + built_size <= len(data):
            uv_log_scales = struct.unpack_from("<I", data, built_offset + 48)[0]
            uv_scale = float(1 << (uv_log_scales & 0xF)) / 16384.0

    positions = values[0:3, :].T.astype(np.float32)
    positions *= np.float32(position_scale)
    positions += np.asarray(position_offset, dtype=np.float32)
    position_ws = values[3, :].astype(np.int32)
    uv0 = values[4:6, :].T.astype(np.float32) * np.float32(uv_scale)

    uv1 = None
    uv1_block = blocks.get(MSMR_MODEL_UV1_VERT_HASH)
    if uv1_block:
        uv1_offset, uv1_size = int(uv1_block[0]), int(uv1_block[1])
        expected_size = vertex_count * 4
        if uv1_offset < 0 or uv1_size < expected_size or uv1_offset + uv1_size > len(data):
            raise ValueError("MSMR Model UV1 Vert does not cover every vertex")
        uv1 = np.frombuffer(data, dtype="<i2", count=vertex_count * 2, offset=uv1_offset)
        uv1 = uv1.reshape(vertex_count, 2).astype(np.float32) / np.float32(32768.0)

    colors = None
    color_words = None
    color_block = blocks.get(MSMR_MODEL_COL_VERT_HASH)
    if color_block:
        color_offset, color_size = int(color_block[0]), int(color_block[1])
        expected_size = vertex_count * 4
        if color_offset < 0 or color_size != expected_size or color_offset + color_size > len(data):
            raise ValueError(
                f"MSMR Model Col Vert has {color_size // 4} colors, expected {vertex_count}"
            )
        colors = np.frombuffer(data, dtype=np.uint8, count=vertex_count * 4, offset=color_offset)
        colors = colors.reshape(vertex_count, 4)
        color_words = np.frombuffer(data, dtype="<u4", count=vertex_count, offset=color_offset)

    if normal_words.shape[0] != vertex_count:
        raise ValueError("MSMR normal and position stream lengths disagree")

    normals = _decode_msmr_normals(normal_words)
    tangents, bitangents, bitangent_signs, tangent_valid = _decode_msmr_tangent_frame(
        normal_words,
        position_ws,
        normals,
    )

    return {
        "indices": indices,
        "positions": positions,
        "normals": normals,
        "tangents": tangents,
        "bitangents": bitangents,
        "bitangent_signs": bitangent_signs,
        "tangent_valid": tangent_valid,
        "normal_words": normal_words,
        "position_ws": position_ws,
        "uv0": uv0,
        "uv1": uv1,
        "colors": colors,
        "color_words": color_words,
        "position_scale": float(position_scale),
        "position_offset": tuple(float(value) for value in position_offset),
    }


def parse_msmr_subset(data, subset_offset, subset_index):
    ptr = int(subset_offset) + int(subset_index) * MSMR_SUBSET_RECORD_SIZE
    if ptr < 0 or ptr + MSMR_SUBSET_RECORD_SIZE > len(data):
        raise ValueError(f"MSMR subset {subset_index} is outside the DAT1 payload")
    vertex_start, index_start, index_count, vertex_count = struct.unpack_from("<IIII", data, ptr + 20)
    flags, material_index, first_skin_batch = struct.unpack_from("<HHH", data, ptr + 36)
    skin_batch_count = data[ptr + 42]
    return {
        "index": int(subset_index),
        "vertex_start": int(vertex_start),
        "index_start": int(index_start),
        "index_count": int(index_count),
        "vertex_count": int(vertex_count),
        "flags": int(flags),
        "material_index": int(material_index),
        "first_skin_batch": int(first_skin_batch),
        "skin_batch_count": int(skin_batch_count),
    }


def msmr_subset_indices(geometry, subset):
    start = int(subset["index_start"])
    count = int(subset["index_count"])
    vertex_start = int(subset["vertex_start"])
    vertex_count = int(subset["vertex_count"])
    indices = geometry["indices"]
    if start < 0 or count < 0 or start > len(indices) or count > len(indices) - start:
        raise ValueError(f"MSMR subset {subset['index']} references indexes outside Model Index")
    result = indices[start:start + count].astype(np.int32)
    if not (int(subset["flags"]) & 0x10):
        result -= vertex_start
    if result.size and (int(result.min()) < 0 or int(result.max()) >= vertex_count):
        raise ValueError(f"MSMR subset {subset['index']} has vertex indexes outside its vertex range")
    return result


def parse_msmr_looks_metadata(data, blocks, read_string):
    look = blocks.get(MSMR_MODEL_LOOK_HASH)
    if not look:
        return []
    look_offset, look_size = int(look[0]), int(look[1])
    if look_size % 32 or look_offset < 0 or look_offset + look_size > len(data):
        raise ValueError("MSMR Model Look has an invalid size")

    subset_block = blocks.get(MSMR_MODEL_SUBSET_HASH, (0, 0))
    subset_count = int(subset_block[1]) // MSMR_SUBSET_RECORD_SIZE
    look_count = look_size // 32
    built = blocks.get(MSMR_MODEL_LOOK_BUILT_HASH)
    built_offset, built_size = (int(built[0]), int(built[1])) if built else (0, 0)

    looks = []
    for look_index in range(look_count):
        ptr = look_offset + look_index * 32
        raw_lods = []
        for lod_index in range(8):
            start, count = struct.unpack_from("<HH", data, ptr + lod_index * 4)
            start = int(start)
            count = int(count)
            if start > subset_count or count > subset_count - start:
                raise ValueError(
                    f"MSMR Model Look {look_index} LOD {lod_index} references subsets outside Model Subset"
                )
            raw_lods.append((start, count))

        name = f"Look {look_index}"
        name_hash = 0
        built_ptr = built_offset + look_index * 80
        if built and built_ptr + 80 <= built_offset + built_size:
            _count, _original_hash, name_hash, name_offset = struct.unpack_from("<4I", data, built_ptr + 64)
            read_name = read_string(data, name_offset)
            if read_name:
                name = read_name

        # MSMR stores global subset ranges directly in Model Look.  Normalize
        # those ranges into the add-on's per-look subset list representation.
        subset_ids = sorted({
            subset_index
            for start, count in raw_lods
            for subset_index in range(start, start + count)
        })
        subset_positions = {value: index for index, value in enumerate(subset_ids)}
        lods = []
        for start, count in raw_lods:
            normalized_start = subset_positions.get(start, 0) if count else 0
            lods.append({"start": normalized_start, "count": count})

        looks.append({
            "index": int(look_index),
            "name": name,
            "name_hash": int(name_hash) & 0xFFFFFFFF,
            "lods": lods,
            "subset_ids": subset_ids,
        })
    return looks


def parse_msmr_look_groups_metadata(data, blocks):
    """Decode named MSMR presets containing sets of Model Look indices."""
    group = blocks.get(MSMR_MODEL_LOOK_GROUP_HASH)
    if not group:
        return []
    group_offset, group_size = int(group[0]), int(group[1])
    if group_size <= 0:
        return []
    if group_offset < 0 or group_offset + group_size > len(data):
        raise ValueError("MSMR Model Look Group block is outside the DAT1 payload")

    group_count = int(data[group_offset])
    records_base = group_offset + 1
    records_size = group_count * MSMR_LOOK_GROUP_RECORD_SIZE
    block_end = group_offset + group_size
    records_end = records_base + records_size
    if records_end > block_end:
        raise ValueError("MSMR Model Look Group record table is truncated")

    look_block = blocks.get(MSMR_MODEL_LOOK_HASH, (0, 0))
    look_count = int(look_block[1]) // 32
    groups = []
    for group_index in range(group_count):
        ptr = records_base + group_index * MSMR_LOOK_GROUP_RECORD_SIZE
        (
            indices_relative,
            unknown_offset,
            preset_look_count,
            unknown_count,
            name_hash,
            name_offset,
        ) = struct.unpack_from("<6I", data, ptr)
        indices_ptr = records_base + int(indices_relative)
        indices_size = int(preset_look_count) * 2
        indices_end = indices_ptr + indices_size
        if preset_look_count and (
            indices_ptr < records_end or indices_ptr > block_end or indices_end > block_end
        ):
            raise ValueError(f"MSMR look-group preset {group_index} has an invalid look-index list")
        look_indices = list(
            struct.unpack_from(f"<{int(preset_look_count)}H", data, indices_ptr)
        ) if preset_look_count else []
        if any(int(value) >= look_count for value in look_indices):
            raise ValueError(f"MSMR look-group preset {group_index} references a missing Model Look")

        groups.append({
            "index": int(group_index),
            "name": _read_msmr_name(data, name_offset, name_hash, "LookGroup"),
            "name_hash": int(name_hash) & 0xFFFFFFFF,
            "look_indices": [int(value) for value in look_indices],
            "source_indices_offset": int(indices_relative),
            "source_unknown_offset": int(unknown_offset),
            "source_unknown_count": int(unknown_count),
        })
    return groups


def decode_msmr_joint_metadata(data, blocks, joint_count, joint_hashes=None):
    """Decode the joint name lookup and compact bidirectional mirror map."""
    joint_count = int(joint_count)
    if joint_count < 0:
        raise ValueError("MSMR joint count cannot be negative")
    if joint_hashes is not None:
        joint_hashes = [int(value) & 0xFFFFFFFF for value in joint_hashes]
        if len(joint_hashes) != joint_count:
            raise ValueError("MSMR joint hash count does not match Model Joint")

    lookup_entries = []
    lookup_by_joint = [None] * joint_count
    lookup_sentinel_present = False
    lookup_block = blocks.get(MSMR_MODEL_JOINT_LOOKUP_HASH)
    if lookup_block:
        lookup_offset, lookup_size = _checked_block(
            data, blocks, MSMR_MODEL_JOINT_LOOKUP_HASH, "Model Joint Lookup"
        )
        if lookup_size % MSMR_JOINT_LOOKUP_RECORD_SIZE:
            raise ValueError("MSMR Model Joint Lookup has a truncated record")
        lookup_count = lookup_size // MSMR_JOINT_LOOKUP_RECORD_SIZE
        previous_hash = -1
        seen_hashes = set()
        for lookup_index in range(lookup_count):
            name_hash, joint_index = struct.unpack_from(
                "<II", data, lookup_offset + lookup_index * MSMR_JOINT_LOOKUP_RECORD_SIZE
            )
            name_hash = int(name_hash)
            joint_index = int(joint_index)
            if name_hash < previous_hash:
                raise ValueError("MSMR Model Joint Lookup is not hash-sorted")
            previous_hash = name_hash
            if name_hash == 0xFFFFFFFF and joint_index == 0xFFFFFFFF:
                if lookup_sentinel_present or lookup_index + 1 != lookup_count:
                    raise ValueError("MSMR Model Joint Lookup has an invalid sentinel")
                lookup_sentinel_present = True
                continue
            if name_hash == 0xFFFFFFFF or joint_index == 0xFFFFFFFF:
                raise ValueError("MSMR Model Joint Lookup has a partial sentinel")
            if joint_index >= joint_count:
                raise ValueError("MSMR Model Joint Lookup references a missing joint")
            if name_hash in seen_hashes or lookup_by_joint[joint_index] is not None:
                raise ValueError("MSMR Model Joint Lookup contains a duplicate mapping")
            if joint_hashes is not None and joint_hashes[joint_index] != name_hash:
                raise ValueError("MSMR Model Joint Lookup hash does not match Model Joint")
            seen_hashes.add(name_hash)
            lookup_by_joint[joint_index] = name_hash
            lookup_entries.append({
                "hash": name_hash,
                "joint_index": joint_index,
            })
        if any(value is None for value in lookup_by_joint):
            raise ValueError("MSMR Model Joint Lookup does not map every joint")

    mirror_records = []
    mirror_by_joint = [None] * joint_count
    mirror_block = blocks.get(MSMR_MODEL_MIRROR_IDS_HASH)
    if mirror_block:
        mirror_offset, mirror_size = _checked_block(
            data, blocks, MSMR_MODEL_MIRROR_IDS_HASH, "Model Mirror Ids"
        )
        if mirror_size % MSMR_MIRROR_ID_RECORD_SIZE:
            raise ValueError("MSMR Model Mirror Ids has a truncated record")
        for record_index in range(mirror_size // MSMR_MIRROR_ID_RECORD_SIZE):
            encoded_source, partner_index = struct.unpack_from(
                "<HH", data, mirror_offset + record_index * MSMR_MIRROR_ID_RECORD_SIZE
            )
            source_index = int(encoded_source) & 0x0FFF
            flags = int(encoded_source) >> 12
            partner_index = int(partner_index)
            if source_index >= joint_count or partner_index >= joint_count:
                raise ValueError("MSMR Model Mirror Ids references a missing joint")
            mirror_records.append({
                "record_index": int(record_index),
                "encoded_source": int(encoded_source),
                "source_joint": source_index,
                "partner_joint": partner_index,
                "flags": flags,
            })
            for joint_index, mirrored_index, is_source in (
                (source_index, partner_index, True),
                (partner_index, source_index, source_index == partner_index),
            ):
                mapping = {
                    "record_index": int(record_index),
                    "record_source": bool(is_source),
                    "partner_joint": int(mirrored_index),
                    "flags": flags,
                }
                existing = mirror_by_joint[joint_index]
                if existing is not None and existing != mapping:
                    raise ValueError("MSMR Model Mirror Ids maps a joint more than once")
                mirror_by_joint[joint_index] = mapping
        if any(value is None for value in mirror_by_joint):
            raise ValueError("MSMR Model Mirror Ids does not cover every joint")
        for joint_index, mapping in enumerate(mirror_by_joint):
            partner = mirror_by_joint[int(mapping["partner_joint"])]
            if partner is None or int(partner["partner_joint"]) != joint_index:
                raise ValueError("MSMR Model Mirror Ids is not bidirectional")

    return {
        "lookup_entries": lookup_entries,
        "lookup_by_joint": lookup_by_joint,
        "lookup_sentinel_present": bool(lookup_sentinel_present),
        "mirror_records": mirror_records,
        "mirror_by_joint": mirror_by_joint,
    }


def decode_msmr_locators(data, blocks):
    """Decode named model-space or joint-relative attachment transforms."""
    locator_block = blocks.get(MSMR_MODEL_LOCATOR_HASH)
    if not locator_block:
        return []
    locator_offset, locator_size = _checked_block(
        data, blocks, MSMR_MODEL_LOCATOR_HASH, "Model Locator"
    )
    if locator_size % MSMR_LOCATOR_RECORD_SIZE:
        raise ValueError("MSMR Model Locator has a truncated record")
    locator_count = locator_size // MSMR_LOCATOR_RECORD_SIZE

    locators = []
    for locator_index in range(locator_count):
        ptr = locator_offset + locator_index * MSMR_LOCATOR_RECORD_SIZE
        name_hash, name_offset, parent_joint, zero = struct.unpack_from("<IIiI", data, ptr)
        raw_rows = np.asarray(struct.unpack_from("<12f", data, ptr + 16), dtype=np.float32).reshape(4, 3)
        if not np.isfinite(raw_rows).all():
            raise ValueError(f"MSMR locator {locator_index} has a non-finite transform")

        matrix = np.identity(4, dtype=np.float32)
        matrix[:3, :3] = raw_rows[:3, :].T
        matrix[:3, 3] = raw_rows[3, :]
        locators.append({
            "index": int(locator_index),
            "name": _read_msmr_name(data, name_offset, name_hash, "Locator"),
            "hash": int(name_hash) & 0xFFFFFFFF,
            "parent_joint": int(parent_joint),
            "zero": int(zero),
            "matrix": matrix,
            "matrix_rows": tuple(tuple(float(value) for value in row) for row in raw_rows),
        })

    lookup_block = blocks.get(MSMR_MODEL_LOCATOR_LOOKUP_HASH)
    if lookup_block:
        lookup_offset, lookup_size = _checked_block(
            data, blocks, MSMR_MODEL_LOCATOR_LOOKUP_HASH, "Model Locator Lookup"
        )
        if lookup_size % MSMR_LOCATOR_LOOKUP_RECORD_SIZE:
            raise ValueError("MSMR Model Locator Lookup has a truncated record")
        if lookup_size // MSMR_LOCATOR_LOOKUP_RECORD_SIZE != locator_count:
            raise ValueError("MSMR locator definition and lookup counts disagree")
        seen_indices = set()
        previous_hash = -1
        for lookup_index in range(locator_count):
            name_hash, locator_index = struct.unpack_from(
                "<II", data, lookup_offset + lookup_index * MSMR_LOCATOR_LOOKUP_RECORD_SIZE
            )
            if locator_index >= locator_count or int(locator_index) in seen_indices:
                raise ValueError("MSMR Model Locator Lookup contains an invalid locator index")
            if locators[int(locator_index)]["hash"] != int(name_hash):
                raise ValueError("MSMR Model Locator Lookup name hash does not match its locator")
            if int(name_hash) < previous_hash:
                raise ValueError("MSMR Model Locator Lookup is not hash-sorted")
            previous_hash = int(name_hash)
            seen_indices.add(int(locator_index))

    return locators


def decode_msmr_splines(data, blocks):
    """Decode MSMR hair/fur spline subsets and their quantized control points."""
    subset_block = blocks.get(MSMR_MODEL_SPLINE_SUBSETS_HASH)
    if not subset_block:
        return None
    subset_offset, subset_size = _checked_block(
        data, blocks, MSMR_MODEL_SPLINE_SUBSETS_HASH, "Model Spline Subsets"
    )
    spline_offset, spline_size = _checked_block(
        data, blocks, MSMR_MODEL_SPLINES_HASH, "Model Splines"
    )
    point_offset, point_size = _checked_block(
        data, blocks, MSMR_MODEL_SPLINE_POINTS_HASH, "Model Spline Points"
    )
    if subset_size % MSMR_SPLINE_SUBSET_RECORD_SIZE:
        raise ValueError("MSMR Model Spline Subsets has a truncated record")
    if spline_size % MSMR_SPLINE_RECORD_SIZE:
        raise ValueError("MSMR Model Splines has a truncated record")
    if point_size % MSMR_SPLINE_POINT_RECORD_SIZE:
        raise ValueError("MSMR Model Spline Points has a truncated record")

    spline_count = spline_size // MSMR_SPLINE_RECORD_SIZE
    spline_dtype = np.dtype([
        ("value_u16", "<u2"),
        ("lane", "u1"),
        ("point_count", "u1"),
        ("packed_0", "<u4"),
        ("packed_1", "<u4"),
    ])
    spline_records = np.frombuffer(
        data, dtype=spline_dtype, count=spline_count, offset=spline_offset
    )
    point_counts = spline_records["point_count"].astype(np.int32)
    if point_counts.size and int(point_counts.min()) < 2:
        raise ValueError("MSMR Model Splines contains a curve with fewer than two points")
    point_offsets = np.empty(spline_count + 1, dtype=np.int64)
    point_offsets[0] = 0
    np.cumsum(point_counts, dtype=np.int64, out=point_offsets[1:])
    point_count = int(point_offsets[-1])
    if point_size != point_count * MSMR_SPLINE_POINT_RECORD_SIZE:
        raise ValueError(
            f"MSMR spline records declare {point_count} points, but the point block contains "
            f"{point_size // MSMR_SPLINE_POINT_RECORD_SIZE}"
        )

    raw_points = np.frombuffer(
        data, dtype="<u2", count=point_count * 4, offset=point_offset
    ).reshape(point_count, 4)
    signed_points = raw_points.view(np.int16).reshape(point_count, 4)
    position_scale, position_offset = _msmr_position_quantization(data, blocks)
    positions = signed_points[:, :3].astype(np.float32)
    positions *= np.float32(position_scale)
    positions += np.asarray(position_offset, dtype=np.float32)

    skin_bindings = None
    binding_block = blocks.get(MSMR_MODEL_SPLINE_SKIN_BINDING_HASH)
    if binding_block:
        binding_offset, binding_size = _checked_block(
            data, blocks, MSMR_MODEL_SPLINE_SKIN_BINDING_HASH, "Model Spline Skin Binding"
        )
        expected_binding_size = spline_count * MSMR_SPLINE_SKIN_BINDING_RECORD_SIZE
        if binding_size != expected_binding_size:
            raise ValueError(
                f"MSMR spline skin binding count is {binding_size // 8}, expected {spline_count}"
            )
        skin_bindings = np.frombuffer(
            data, dtype="<u2", count=spline_count * 4, offset=binding_offset
        ).reshape(spline_count, 4)

    subset_count = subset_size // MSMR_SPLINE_SUBSET_RECORD_SIZE
    covered_splines = np.zeros(spline_count, dtype=np.bool_)
    subsets = []
    for subset_index in range(subset_count):
        ptr = subset_offset + subset_index * MSMR_SPLINE_SUBSET_RECORD_SIZE
        name_hash, name_offset, count, first_spline = struct.unpack_from("<4I", data, ptr)
        unknown_floats = struct.unpack_from("<2f", data, ptr + 16)
        unknown_fields = struct.unpack_from("<HHBBH", data, ptr + 24)
        count = int(count)
        first_spline = int(first_spline)
        if first_spline > spline_count or count > spline_count - first_spline:
            raise ValueError(f"MSMR spline subset {subset_index} references curves outside Model Splines")
        if covered_splines[first_spline:first_spline + count].any():
            raise ValueError(f"MSMR spline subset {subset_index} overlaps another subset")
        covered_splines[first_spline:first_spline + count] = True

        string_offsets = struct.unpack_from("<5I", data, ptr + 0x468)
        config_offset = struct.unpack_from("<I", data, ptr + 0x490)[0]
        paths = tuple(_read_msmr_optional_string(data, value) for value in string_offsets)
        subsets.append({
            "index": int(subset_index),
            "name": _read_msmr_name(data, name_offset, name_hash, "SplineSubset"),
            "hash": int(name_hash) & 0xFFFFFFFF,
            "first_spline": first_spline,
            "spline_count": count,
            "unknown_floats": tuple(float(value) for value in unknown_floats),
            "unknown_fields": tuple(int(value) for value in unknown_fields),
            "fur_tint_texture": paths[0],
            "string_paths": paths,
            "fur_mask_texture": paths[4],
            "config_path": _read_msmr_optional_string(data, config_offset),
        })
    if spline_count and not covered_splines.all():
        raise ValueError("MSMR spline subsets do not account for every Model Splines record")

    return {
        "subsets": subsets,
        "spline_count": int(spline_count),
        "point_count": int(point_count),
        "point_counts": point_counts,
        "point_offsets": point_offsets,
        "positions": positions,
        "point_w": raw_points[:, 3],
        "value_u16": spline_records["value_u16"],
        "lane": spline_records["lane"],
        "packed_0": spline_records["packed_0"],
        "packed_1": spline_records["packed_1"],
        "skin_bindings": skin_bindings,
        "position_scale": float(position_scale),
        "position_offset": position_offset,
        "has_joint_bindings": MSMR_MODEL_SPLINE_JOINT_BINDING_HASH in blocks,
        "has_joint_weights": MSMR_MODEL_SPLINE_JOINT_WEIGHTS_HASH in blocks,
    }


def has_msmr_morphs(blocks):
    """Return whether all three legacy morph stream blocks are present."""
    return all(
        block_hash in blocks
        for block_hash in (
            MSMR_MODEL_ANIM_MORPH_INFO_HASH,
            MSMR_MODEL_ANIM_MORPH_DATA_HASH,
            MSMR_MODEL_ANIM_MORPH_INDICES_HASH,
        )
    )


def _align4(value):
    return (int(value) + 3) & ~3


def _checked_range(start, size, lower, upper, label):
    start = int(start)
    size = int(size)
    if start < int(lower) or size < 0 or start > int(upper) or size > int(upper) - start:
        raise ValueError(f"MSMR {label} is outside its block")
    return start, start + size


def _read_msmr_name(data, offset, name_hash, fallback_prefix):
    offset = int(offset)
    if not 0 <= offset < len(data):
        return f"{fallback_prefix}_{int(name_hash) & 0xFFFFFFFF:08X}"
    end = data.find(b"\x00", offset, min(len(data), offset + 4096))
    if end < 0:
        end = min(len(data), offset + 4096)
    name = data[offset:end].decode("ascii", errors="ignore")
    return name or f"{fallback_prefix}_{int(name_hash) & 0xFFFFFFFF:08X}"


def _read_msmr_optional_string(data, offset):
    offset = int(offset)
    if offset <= 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset, min(len(data), offset + 4096))
    if end < 0:
        end = min(len(data), offset + 4096)
    return data[offset:end].decode("ascii", errors="ignore")


def _decode_msmr_morph_values(data, offset, vertex_count, component_bits, limit):
    value_count = int(vertex_count) * 3
    required_bits = value_count * int(component_bits)
    byte_count = (required_bits + 7) // 8
    _checked_range(offset, byte_count, 0, limit, "morph delta page")
    packed = np.frombuffer(data, dtype=np.uint8, count=byte_count, offset=int(offset))
    bits = np.unpackbits(packed, bitorder="big")[:required_bits]
    if bits.size != required_bits:
        raise ValueError("MSMR morph delta page is truncated")
    bit_weights = np.left_shift(
        np.uint32(1),
        np.arange(int(component_bits) - 1, -1, -1, dtype=np.uint32),
    )
    values = bits.reshape(value_count, int(component_bits)).dot(bit_weights)
    return values.reshape(int(vertex_count), 3), byte_count


def decode_msmr_morphs(data, blocks, allowed_subset_ids=None):
    """Decode Remastered's legacy morph streams into position shape-key deltas.

    The legacy format stores each target as independently quantized XYZ values,
    split into pages of at most 0xA00 local vertices.  Changed vertex indexes are
    encoded separately as skip/read runs.  Normal scale/bias fields are retained
    as metadata, but Blender shape keys only consume the position channel.
    """
    if not has_msmr_morphs(blocks):
        return None

    info_offset, info_size = _checked_block(
        data, blocks, MSMR_MODEL_ANIM_MORPH_INFO_HASH, "Model Anim Morph Info"
    )
    delta_offset, delta_size = _checked_block(
        data, blocks, MSMR_MODEL_ANIM_MORPH_DATA_HASH, "Model Anim Morph Data"
    )
    index_offset, index_size = _checked_block(
        data, blocks, MSMR_MODEL_ANIM_MORPH_INDICES_HASH, "Model Anim Morph Indices"
    )
    info_end = info_offset + info_size
    delta_end = delta_offset + delta_size
    index_end = index_offset + index_size
    if info_size < 24:
        raise ValueError("MSMR Model Anim Morph Info header is truncated")

    _unknown, buffers_length, target_count, mirror_count, table_relative, mirror_relative, version = (
        struct.unpack_from("<IIHHIII", data, info_offset)
    )
    if int(target_count) > 4096:
        raise ValueError(f"MSMR morph target count {target_count} is unreasonable")
    table_offset = info_offset + int(table_relative)
    mirror_offset = info_offset + int(mirror_relative)
    _checked_range(table_offset, int(target_count) * 8, info_offset, info_end, "morph target table")
    _checked_range(mirror_offset, int(mirror_count) * 8, info_offset, info_end, "morph mirror table")

    allowed = None if allowed_subset_ids is None else {int(value) for value in allowed_subset_ids}
    subset_block_offset, subset_block_size = _checked_block(
        data, blocks, MSMR_MODEL_SUBSET_HASH, "Model Subset"
    )
    subset_count = subset_block_size // MSMR_SUBSET_RECORD_SIZE
    subset_vertex_counts = {
        subset_index: parse_msmr_subset(data, subset_block_offset, subset_index)["vertex_count"]
        for subset_index in range(subset_count)
    }

    targets = []
    for target_index in range(int(target_count)):
        table_name_hash, target_relative = struct.unpack_from("<II", data, table_offset + target_index * 8)
        target_offset = info_offset + int(target_relative)
        _checked_range(target_offset, 48, info_offset, info_end, f"morph target {target_index} header")
        (
            name_hash,
            name_offset,
            target_delta_relative,
            target_index_relative,
            packing_kind,
            packed_vertex_bits,
            component_bits,
            packing_null,
            position_scale,
            position_bias,
            normal_scale,
            normal_bias,
            target_subset_count,
            subset_info_length,
            target_delta_length,
            target_index_length,
        ) = struct.unpack_from("<4I4B4fHHII", data, target_offset)
        if name_hash != table_name_hash:
            raise ValueError(f"MSMR morph target {target_index} name hashes disagree")
        if not 1 <= int(component_bits) <= 24 or int(packed_vertex_bits) != int(component_bits) * 3:
            raise ValueError(f"MSMR morph target {target_index} has unsupported position packing")
        if not math.isfinite(position_scale) or not math.isfinite(position_bias):
            raise ValueError(f"MSMR morph target {target_index} has non-finite position quantization")
        if int(target_subset_count) > subset_count:
            raise ValueError(f"MSMR morph target {target_index} has too many subset records")

        target_delta_start = delta_offset + int(target_delta_relative)
        target_index_start = index_offset + int(target_index_relative)
        target_delta_stop = target_delta_start + int(target_delta_length)
        target_index_stop = target_index_start + int(target_index_length)
        _checked_range(
            target_delta_start,
            target_delta_length,
            delta_offset,
            delta_end,
            f"morph target {target_index} delta payload",
        )
        _checked_range(
            target_index_start,
            target_index_length,
            index_offset,
            index_end,
            f"morph target {target_index} index payload",
        )

        array_offset = target_offset + 48
        ids_start, ids_end = _checked_range(
            array_offset,
            target_subset_count,
            target_offset,
            info_end,
            f"morph target {target_index} subset IDs",
        )
        subset_ids = tuple(int(value) for value in data[ids_start:ids_end])
        if len(set(subset_ids)) != len(subset_ids):
            raise ValueError(f"MSMR morph target {target_index} repeats a subset ID")
        array_offset = _align4(ids_end)

        arrays = []
        for field_name, item_size, format_char in (
            ("delta offsets", 4, "I"),
            ("index offsets", 4, "I"),
            ("vertex counts", 2, "H"),
            ("table indexes", 2, "H"),
        ):
            array_size = int(target_subset_count) * item_size
            start, end = _checked_range(
                array_offset,
                array_size,
                target_offset,
                info_end,
                f"morph target {target_index} subset {field_name}",
            )
            arrays.append(
                struct.unpack_from(f"<{int(target_subset_count)}{format_char}", data, start)
                if target_subset_count
                else ()
            )
            array_offset = _align4(end) if item_size == 2 else end
        subset_delta_offsets, subset_index_offsets, changed_counts, table_indexes = arrays
        data_table_offset = array_offset
        # The field is measured from the start of the subset-ID array, not
        # from the start of the target header.
        declared_table_end = target_offset + 48 + int(subset_info_length)
        if subset_info_length and not data_table_offset <= declared_table_end <= info_end:
            raise ValueError(f"MSMR morph target {target_index} has an invalid subset info length")

        decoded_subsets = []
        for morph_subset_index, subset_id in enumerate(subset_ids):
            if subset_id >= subset_count:
                raise ValueError(f"MSMR morph target {target_index} references subset {subset_id}")
            changed_count = int(changed_counts[morph_subset_index])
            table_ptr = data_table_offset + int(table_indexes[morph_subset_index]) * 6
            pages = []
            page_vertex_total = 0
            while page_vertex_total < changed_count:
                _checked_range(
                    table_ptr,
                    6,
                    data_table_offset,
                    declared_table_end if subset_info_length else info_end,
                    f"morph target {target_index} data table",
                )
                page_vertex_count, run_count = struct.unpack_from("<HI", data, table_ptr)
                table_ptr += 6
                if page_vertex_count == 0:
                    if run_count:
                        raise ValueError(f"MSMR morph target {target_index} has an invalid empty data page")
                    continue
                if page_vertex_total + int(page_vertex_count) > changed_count:
                    raise ValueError(f"MSMR morph target {target_index} has an invalid page vertex count")
                pages.append((int(page_vertex_count), int(run_count)))
                page_vertex_total += int(page_vertex_count)

            if allowed is not None and subset_id not in allowed:
                continue

            delta_ptr = target_delta_start + int(subset_delta_offsets[morph_subset_index])
            run_ptr = target_index_start + int(subset_index_offsets[morph_subset_index])
            deltas = {}
            for page_index, (page_vertex_count, run_count) in enumerate(pages):
                quantized, byte_count = _decode_msmr_morph_values(
                    data,
                    delta_ptr,
                    page_vertex_count,
                    component_bits,
                    target_delta_stop,
                )
                page_deltas = (
                    quantized.astype(np.float32) * np.float32(position_scale) + np.float32(position_bias)
                )
                delta_ptr = _align4(delta_ptr + byte_count)

                _checked_range(
                    run_ptr,
                    run_count * 4,
                    target_index_start,
                    target_index_stop,
                    f"morph target {target_index} index page",
                )
                vertex_indices = []
                current_vertex = page_index * MSMR_MORPH_PAGE_VERTEX_COUNT
                for run_index in range(run_count):
                    skip_count, read_count = struct.unpack_from("<HH", data, run_ptr + run_index * 4)
                    read_count = int(read_count) or 0x20
                    current_vertex += int(skip_count)
                    vertex_indices.extend(range(current_vertex, current_vertex + read_count))
                    current_vertex += read_count
                run_ptr += run_count * 4
                if len(vertex_indices) != page_vertex_count:
                    raise ValueError(
                        f"MSMR morph target {target_index} index page expands to "
                        f"{len(vertex_indices)} vertices, expected {page_vertex_count}"
                    )
                subset_vertex_count = int(subset_vertex_counts[subset_id])
                if vertex_indices and vertex_indices[-1] >= subset_vertex_count:
                    raise ValueError(
                        f"MSMR morph target {target_index} references vertex {vertex_indices[-1]} "
                        f"outside subset {subset_id}"
                    )
                for vertex_index, delta in zip(vertex_indices, page_deltas):
                    if int(vertex_index) in deltas:
                        raise ValueError(
                            f"MSMR morph target {target_index} repeats vertex {vertex_index} in subset {subset_id}"
                        )
                    deltas[int(vertex_index)] = (float(delta[0]), float(delta[1]), float(delta[2]))

            if len(deltas) != changed_count:
                raise ValueError(
                    f"MSMR morph target {target_index} decoded {len(deltas)} unique vertices, "
                    f"expected {changed_count}"
                )

            decoded_subsets.append({"subset_index": subset_id, "deltas": deltas})

        targets.append({
            "name": _read_msmr_name(data, name_offset, name_hash, "Morph"),
            "hash": int(name_hash) & 0xFFFFFFFF,
            "index": target_index,
            "packing_kind": int(packing_kind),
            "packing_null": int(packing_null),
            "normal_scale": float(normal_scale),
            "normal_bias": float(normal_bias),
            "subsets": decoded_subsets,
        })

    mirrors = [
        tuple(int(value) & 0xFFFFFFFF for value in struct.unpack_from("<II", data, mirror_offset + i * 8))
        for i in range(int(mirror_count))
    ]
    return {
        "target_count": int(target_count),
        "targets": targets,
        "mirrors": mirrors,
        "buffers_length": int(buffers_length),
        "version": int(version),
        "format": "MSMR_LEGACY",
    }


def decode_msmr_skin_weights(data, blocks, subset, joint_count):
    """Return joint -> encoded weight -> local vertex indexes for one subset."""
    first_batch = int(subset.get("first_skin_batch", 0))
    batch_count = int(subset.get("skin_batch_count", 0))
    vertex_count = int(subset.get("vertex_count", 0))
    if batch_count <= 0 or vertex_count <= 0:
        return {}

    batch_offset, batch_size = _checked_block(data, blocks, MSMR_MODEL_SKIN_BATCH_HASH, "Model Skin Batch")
    skin_offset, skin_size = _checked_block(data, blocks, MSMR_MODEL_SKIN_DATA_HASH, "Model Skin Data")
    total_batches = batch_size // MSMR_SKIN_BATCH_RECORD_SIZE
    if batch_size % MSMR_SKIN_BATCH_RECORD_SIZE or first_batch + batch_count > total_batches:
        raise ValueError(f"MSMR subset {subset['index']} has an invalid skin batch range")

    remap_block = blocks.get(MSMR_MODEL_SKIN_JOINT_REMAP_HASH)
    remap_offset, remap_size = (int(remap_block[0]), int(remap_block[1])) if remap_block else (0, 0)
    if remap_block and (
        remap_offset < 0
        or remap_size < 0
        or remap_offset > len(data)
        or remap_size > len(data) - remap_offset
    ):
        raise ValueError("MSMR Model Skin Joint Remap block is outside the DAT1 payload")

    grouped = {}
    for batch_index in range(first_batch, first_batch + batch_count):
        ptr = batch_offset + batch_index * MSMR_SKIN_BATCH_RECORD_SIZE
        data_offset, joint_remap_offset, joint_remap_count, _unknown, count, first_vertex = struct.unpack_from(
            "<IIHHHH", data, ptr
        )
        data_ptr = skin_offset + int(data_offset)
        data_end = skin_offset + skin_size
        joint_remap = None
        if joint_remap_count:
            joint_remap_end = int(joint_remap_offset) + int(joint_remap_count) * 2
            if not remap_block or joint_remap_end > remap_size:
                raise ValueError(f"MSMR skin batch {batch_index} has an invalid joint remap range")
            joint_remap = np.frombuffer(
                data,
                dtype="<u2",
                count=int(joint_remap_count),
                offset=remap_offset + int(joint_remap_offset),
            )
        processed = 0
        while processed < int(count):
            if data_ptr >= data_end:
                raise ValueError(f"MSMR skin batch {batch_index} ends outside Model Skin Data")
            influence_count = int(data[data_ptr]) + 1
            data_ptr += 1
            vertices_in_group = min(MSMR_SKIN_BATCH_VERTEX_COUNT, int(count) - processed)
            for group_vertex in range(vertices_in_group):
                local_vertex = int(first_vertex) + processed + group_vertex
                if local_vertex >= vertex_count:
                    raise ValueError(f"MSMR skin batch {batch_index} exceeds subset {subset['index']}")
                if influence_count == 1:
                    if data_ptr >= data_end:
                        raise ValueError(f"MSMR skin batch {batch_index} is truncated")
                    influences = ((int(data[data_ptr]), 256),)
                    data_ptr += 1
                else:
                    byte_count = influence_count * 2
                    if data_ptr + byte_count > data_end:
                        raise ValueError(f"MSMR skin batch {batch_index} is truncated")
                    influences = tuple(
                        (int(data[data_ptr + influence * 2]), int(data[data_ptr + influence * 2 + 1]))
                        for influence in range(influence_count)
                    )
                    data_ptr += byte_count
                combined_influences = {}
                for encoded_joint_index, weight in influences:
                    if weight <= 0:
                        continue
                    if joint_remap is not None:
                        if encoded_joint_index >= len(joint_remap):
                            raise ValueError(
                                f"MSMR skin batch {batch_index} references joint remap index "
                                f"{encoded_joint_index} outside its palette"
                            )
                        joint_index = int(joint_remap[encoded_joint_index])
                    else:
                        joint_index = encoded_joint_index
                    if joint_index >= int(joint_count):
                        continue
                    combined_influences[joint_index] = combined_influences.get(joint_index, 0) + weight
                for joint_index, weight in combined_influences.items():
                    grouped.setdefault(joint_index, {}).setdefault(weight, []).append(local_vertex)
            processed += vertices_in_group
    return grouped
