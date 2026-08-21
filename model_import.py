# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *
from .model_morph import decode_model_morph2, sync_armature_morph_controls
from .model_msmr import (
    MSMR_MODEL_STD_VERT_HASH,
    MSMR_SUBSET_RECORD_SIZE,
    decode_msmr_geometry,
    decode_msmr_joint_metadata,
    decode_msmr_locators,
    decode_msmr_morphs,
    decode_msmr_splines,
    decode_msmr_skin_weights,
    has_msmr_morphs,
    is_msmr_model,
    msmr_subset_indices,
    parse_msmr_look_groups_metadata,
    parse_msmr_looks_metadata,
    parse_msmr_subset,
)
from .model_ziva import ZIVA_INVALID_ELEM, load_ziva_model, transfer_ziva_channels_to_objects

MODEL_BIND_POSE_FLOATS_PER_JOINT = 12
MODEL_JOINT_RECORD_SIZE = 16
MODEL_JOINT_NAME_BYTES = 256
MODEL_SUBSET_RECORD_SIZE = 128
MODEL_SUBSET_MPU_OFFSET = 24
MODEL_SUBSET_INDEX_COUNT_OFFSET = 0
MODEL_SUBSET_VERTEX_COUNT_OFFSET = 4
MODEL_SUBSET_INDEX_DATA_OFFSET = 12
MODEL_SUBSET_FLAGS_OFFSET = 20
MODEL_SUBSET_UV_LOG_OFFSET = 22
MODEL_SUBSET_MATERIAL_INDEX_OFFSET = 28
MODEL_SUBSET_VERTEX_STD_OFFSET = 64
MODEL_SUBSET_VERTEX_UV12_OFFSET = 68
MODEL_SUBSET_SKIN_OFFSET = 76
MODEL_SUBSET_CLUSTER_OFFSET = 80
MODEL_SUBSET_BASE_OFFSET = 88
MODEL_SUBSET_LOD_MASK_OFFSET = 102
MODEL_VERTEX_SHORTS_PER_VERTEX = 8
MODEL_MATERIAL_INFO_SIZE = 16
MODEL_MATERIAL_SIZE = 16
MODEL_LOOK_SIZE = 64
MODEL_LOOK_BUILT_SIZE = 80
MODEL_LOOK_GROUP_SIZE = 24
MODEL_BUILT_BSPHERE_OFFSET = 32
MODEL_BUILT_AABB_EXTENTS_OFFSET = 48
MODEL_BUILT_COMMON_MPU_OFFSET = 60
MODEL_BUILT_VERTEX_MPU_OFFSET = 64
MODEL_UV_FLOAT_TO_FIXED_BASE = 16384.0
TRIANGLE_INDEX_COUNT = 3
SUBSET_FLAG_SKINNED = 0x0001
SUBSET_FLAG_HAS_UV1 = 0x0004
SUBSET_FLAG_HAS_UV2 = 0x0008
SKIN_CLUSTER_VERTEX_COUNT = 32
SKIN_CLUSTER_WORD_BYTES = 4
SKIN_CLUSTER_OFFSET_MASK = 0xFFFFF
SKIN_CLUSTER_FULL_INDEX_BIT = 1 << 20
SKIN_CLUSTER_INFLUENCE_SHIFT = 21
SKIN_CLUSTER_JOINT_OFFSET_SHIFT = 25
SKIN_CLUSTER_NIBBLE_MASK = 0xF
SKIN_WEIGHT_SCALE = 256.0

def _read_c_string(data, offset, max_len=4096):
    offset = int(offset)
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b'\x00', offset, min(len(data), offset + max_len))
    if end < 0:
        end = min(len(data), offset + max_len)
    return data[offset:end].decode('ascii', errors='ignore')


def _uv_fixed_to_float(uv_log_scales, channel):
    shift = (int(uv_log_scales) >> (channel * 4)) & 0xF
    return float(1 << shift) / MODEL_UV_FLOAT_TO_FIXED_BASE


def _engine_uv_to_blender(uv):
    return (float(uv[0]), 1.0 - float(uv[1]))


def _decode_azimuthal(x, y, z_sign):
    enc_x = float(x) * (4.0 / 1.41421356) - (2.0 / 1.41421356)
    enc_y = float(y) * (4.0 / 1.41421356) - (2.0 / 1.41421356)
    f = enc_x * enc_x + enc_y * enc_y
    xy_scale = math.sqrt(max(0.0, 1.0 - f * 0.25))
    z = abs(1.0 - f * 0.5)
    if float(z_sign) < 0.5:
        z = -z
    return (enc_x * xy_scale, enc_y * xy_scale, z)


def _decode_packed_normal(word):
    word = int(word) & U32_MASK
    x = float(word & 0x3FF) / 1023.0
    y = float((word >> 10) & 0x3FF) / 1023.0
    alpha = float((word >> 30) & 0x3) / 3.0
    normal_z = max(0.0, min(1.0, alpha * 3.0 - 1.0))
    return _decode_azimuthal(x, y, normal_z)


def _decode_packed_tangent(word, position_w):
    word = int(word) & U32_MASK
    tangent_x = float((word >> 20) & 0x3FF) / 1023.0
    tangent_y = float(abs(int(position_w)) & 0x3FF) / 1023.0
    nt_zs = float((word >> 30) & 0x3)
    normal_z = max(0.0, min(1.0, nt_zs - 1.0))
    tangent_z = nt_zs - normal_z * 2.0
    return _decode_azimuthal(tangent_x, tangent_y, tangent_z)


def _engine_delta_to_blender(delta):
    return (float(delta[0]), -float(delta[2]), float(delta[1]))


def _import_morph_shape_keys(morph, subset_objects, arm, wm=None):
    if not morph:
        return 0
    metadata_by_object = {}
    imported_count = 0
    total_targets = max(1, len(morph.get("targets", [])))
    for target_index, target in enumerate(morph.get("targets", [])):
        for subset in target.get("subsets", []):
            obj = subset_objects.get(int(subset.get("subset_index", -1)))
            if obj is None:
                continue
            mesh = obj.data
            if mesh.shape_keys is None or not mesh.shape_keys.key_blocks:
                obj.shape_key_add(name="Basis", from_mix=False)
            key = obj.shape_key_add(name=str(target.get("name", "Morph")), from_mix=False)
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            for vertex_index, engine_delta in subset.get("deltas", {}).items():
                vertex_index = int(vertex_index)
                if not 0 <= vertex_index < len(mesh.vertices):
                    raise ValueError(
                        f"Morph target {target.get('name', target_index)} references vertex {vertex_index} "
                        f"outside subset {subset.get('subset_index')}"
                    )
                delta = _engine_delta_to_blender(engine_delta)
                base = vertex_index * 3
                coords[base] += delta[0]
                coords[base + 1] += delta[1]
                coords[base + 2] += delta[2]
            key.data.foreach_set("co", coords)
            key.value = 0.0
            metadata_by_object.setdefault(obj, {})[key.name] = {
                "name": str(target.get("name", key.name)),
                "hash": int(target.get("hash", 0)) & U32_MASK,
                "index": int(target.get("index", target_index)),
            }
            imported_count += 1
        if wm and (target_index & 15) == 0:
            wm.progress_update(95 + int(4 * (target_index / total_targets)))
    for obj, metadata in metadata_by_object.items():
        obj["engine_morph_targets_json"] = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        obj["engine_morph_shape_keys_imported"] = True
    arm["engine_model_shape_keys_imported"] = True
    arm["engine_model_morph_target_count"] = int(morph.get("target_count", 0))
    return imported_count


def _parse_model_materials(data, blocks):
    material_block = blocks.get(BLOCK_HASHES.get("ModelMaterial"))
    if not material_block:
        return []
    mat_off, mat_size = material_block
    record_size = MODEL_MATERIAL_INFO_SIZE + MODEL_MATERIAL_SIZE
    material_count = mat_size // record_size
    materials = []
    runtime_base = mat_off + material_count * MODEL_MATERIAL_INFO_SIZE
    for index in range(material_count):
        info_ptr = mat_off + index * MODEL_MATERIAL_INFO_SIZE
        runtime_ptr = runtime_base + index * MODEL_MATERIAL_SIZE
        asset_name_offset, _pad0, mapping_name_offset, _pad1 = struct.unpack_from("<IIII", data, info_ptr)
        material_id, mapping_hash, flags = struct.unpack_from("<QII", data, runtime_ptr)
        asset_path = _read_c_string(data, asset_name_offset)
        mapping_name = _read_c_string(data, mapping_name_offset)
        materials.append({
            "index": index,
            "path": asset_path,
            "mapping": mapping_name,
            "id": material_id,
            "mapping_hash": mapping_hash,
            "flags": flags,
        })
    return materials


def _material_display_name(material_info):
    path = str(material_info.get("path", "") or "")
    if path:
        return path
    mapping = str(material_info.get("mapping", "") or "")
    if mapping:
        return mapping if mapping.lower().endswith(".material") else f"{mapping}.material"
    return f"Engine_Material_{int(material_info.get('index', 0)):03d}.material"


def _make_import_material(material_info):
    path = str(material_info.get("path", "") or "")
    if path and not path.lower().endswith(".material"):
        path = f"{path}.material"
    display_name = _material_display_name({**material_info, "path": path})
    if path:
        for existing in bpy.data.materials:
            if str(getattr(existing, "engine_material_path", "") or "") == path:
                existing.name = display_name
                mat = existing
                break
        else:
            mat = bpy.data.materials.new(display_name)
    else:
        mat = bpy.data.materials.new(display_name)
    mapping = str(material_info.get("mapping", "") or "")
    mat.engine_material_path = path
    mat.engine_material_mapping_name = mapping
    mat["engine_material_index"] = int(material_info.get("index", 0))
    mat["engine_material_id"] = f"0x{int(material_info.get('id', 0)) & U64_MASK:016X}"
    mat["engine_material_mapping_hash"] = to_signed_32(material_info.get("mapping_hash", 0))
    mat["engine_material_flags"] = to_signed_32(material_info.get("flags", 0))
    return mat


def _model_block_name_map():
    names = {}
    for name, value in BLOCK_HASHES.items():
        names[int(value) & U32_MASK] = name
    return names


def _parse_model_blocks_metadata(data, blocks):
    names = _model_block_name_map()
    result = []
    for block_hash, (offset, size) in sorted(blocks.items(), key=lambda item: (item[1][0], item[0])):
        result.append({
            "hash": int(block_hash) & U32_MASK,
            "name": names.get(int(block_hash) & U32_MASK, f"0x{int(block_hash) & U32_MASK:08X}"),
            "offset": int(offset),
            "size": int(size),
        })
    return result


def _parse_model_looks_metadata(data, blocks):
    if is_msmr_model(blocks):
        return parse_msmr_looks_metadata(data, blocks, _read_c_string)
    look_block = blocks.get(BLOCK_HASHES.get("ModelLook"))
    built_block = blocks.get(BLOCK_HASHES.get("ModelLookBuilt"))
    if not look_block:
        return []
    look_off, look_size = look_block
    look_count = look_size // MODEL_LOOK_SIZE
    built_off, built_size = built_block if built_block else (0, 0)
    looks = []
    for look_index in range(look_count):
        ptr = look_off + look_index * MODEL_LOOK_SIZE
        lods = []
        for lod_index in range(8):
            start, count = struct.unpack_from("<HH", data, ptr + lod_index * 4)
            lods.append({"start": int(start), "count": int(count)})

        name = f"Look {look_index}"
        name_hash = 0
        subset_ids = []
        built_ptr = built_off + look_index * MODEL_LOOK_BUILT_SIZE
        if built_block and built_ptr + MODEL_LOOK_BUILT_SIZE <= built_off + built_size:
            subset_ids_offset = struct.unpack_from("<Q", data, built_ptr)[0]
            subset_id_count = struct.unpack_from("<H", data, built_ptr + 56)[0]
            name_hash, _name_hash_lower, name_offset = struct.unpack_from("<3I", data, built_ptr + 68)
            read_name = _read_c_string(data, name_offset)
            if read_name:
                name = read_name
            ids_ptr = built_off + int(subset_ids_offset)
            ids_end = ids_ptr + int(subset_id_count) * 2
            if built_off <= ids_ptr <= ids_end <= built_off + built_size and subset_id_count:
                subset_ids = list(struct.unpack_from(f"<{int(subset_id_count)}H", data, ids_ptr))

        looks.append({
            "index": int(look_index),
            "name": name,
            "name_hash": int(name_hash) & U32_MASK,
            "lods": lods,
            "subset_ids": [int(value) for value in subset_ids],
        })
    return looks


def _collect_model_lod_subset_ids(looks, lod_index):
    ids = set()
    for look in looks or []:
        subset_ids = look.get("subset_ids", [])
        lods = look.get("lods", [])
        if not isinstance(subset_ids, list) or not isinstance(lods, list):
            continue
        try:
            lod = lods[max(0, min(int(lod_index), len(lods) - 1))]
            start = int(lod.get("start", 0))
            count = int(lod.get("count", 0))
        except Exception:
            continue
        if start < 0 or count <= 0:
            continue
        for value in subset_ids[start:start + count]:
            try:
                ids.add(int(value))
            except Exception:
                pass
    return ids


def _parse_model_look_groups_metadata(data, blocks):
    if is_msmr_model(blocks):
        return parse_msmr_look_groups_metadata(data, blocks)
    group_block = blocks.get(BLOCK_HASHES.get("ModelLookGroup"))
    if not group_block:
        return []
    group_off, group_size = group_block
    if group_size <= 0:
        return []
    group_count = data[group_off]
    records_base = group_off + 1
    groups = []
    for group_index in range(min(int(group_count), max(0, (group_size - 1) // MODEL_LOOK_GROUP_SIZE))):
        ptr = records_base + group_index * MODEL_LOOK_GROUP_SIZE
        if ptr + MODEL_LOOK_GROUP_SIZE > group_off + group_size:
            break
        indices_offset, look_count = struct.unpack_from("<QH", data, ptr)
        name_hash, name_offset = struct.unpack_from("<II", data, ptr + 16)
        name = _read_c_string(data, name_offset) or f"Look Group {group_index}"
        indices_ptr = records_base + int(indices_offset)
        indices_end = indices_ptr + int(look_count) * 2
        look_indices = []
        if records_base <= indices_ptr <= indices_end <= group_off + group_size and look_count:
            look_indices = list(struct.unpack_from(f"<{int(look_count)}H", data, indices_ptr))
        groups.append({
            "index": int(group_index),
            "name": name,
            "name_hash": int(name_hash) & U32_MASK,
            "look_indices": [int(value) for value in look_indices],
        })
    return groups


def _store_model_metadata(arm, filepath, data, blocks, sb_offset, source_had_stg, dat1_version, joint_count, material_count):
    block_count = struct.unpack_from("<H", data, 12)[0] if len(data) >= 14 else len(blocks)
    fixup_count = struct.unpack_from("<H", data, 14)[0] if len(data) >= 16 else 0
    msmr = is_msmr_model(blocks)
    geom_block = blocks.get(
        MSMR_MODEL_STD_VERT_HASH if msmr else BLOCK_HASHES.get("ModelSubsetGeomData"),
        (0, 0),
    )
    subset_block = blocks.get(BLOCK_HASHES.get("ModelSubset"), (0, 0))
    first_block_offset = min((offset for offset, size in blocks.values() if size or offset), default=sb_offset)

    arm["engine_mpu"] = float(arm.get("engine_mpu", 1.0))
    arm["engine_model_joint_count"] = int(joint_count)
    arm["engine_model_source_path"] = os.path.abspath(filepath)
    arm["engine_model_source_had_stg"] = bool(source_had_stg)
    arm["engine_model_dat1_version"] = to_signed_32(dat1_version)
    arm["engine_model_dat1_size"] = int(len(data))
    arm["engine_model_block_count"] = int(block_count)
    arm["engine_model_fixup_count"] = int(fixup_count)
    arm["engine_model_string_table_size"] = max(0, int(first_block_offset) - int(sb_offset))
    arm["engine_model_material_count"] = int(material_count)
    subset_record_size = MSMR_SUBSET_RECORD_SIZE if msmr else MODEL_SUBSET_RECORD_SIZE
    arm["engine_model_subset_count"] = int(subset_block[1] // subset_record_size) if subset_block else 0
    arm["engine_model_geom_offset"] = int(geom_block[0])
    arm["engine_model_geom_size"] = int(geom_block[1])
    arm["engine_model_format"] = "MSMR" if msmr else "MSM2"
    looks_metadata = _parse_model_looks_metadata(data, blocks)
    look_groups_metadata = _parse_model_look_groups_metadata(data, blocks)
    arm["engine_model_blocks_json"] = json.dumps(_parse_model_blocks_metadata(data, blocks), separators=(",", ":"))
    arm["engine_model_looks_json"] = json.dumps(looks_metadata, separators=(",", ":"))
    arm["engine_model_look_groups_json"] = json.dumps(look_groups_metadata, separators=(",", ":"))
    arm["engine_model_look_count"] = int(len(looks_metadata))
    arm["engine_model_look_group_count"] = int(len(look_groups_metadata))

    model_built = blocks.get(BLOCK_HASHES.get("ModelBuilt"))
    if model_built and msmr:
        mb_offset, mb_size = model_built
        if mb_size >= 52:
            position_offset = struct.unpack_from("<3f", data, mb_offset + 28)
            vertex_mpu = struct.unpack_from("<f", data, mb_offset + 44)[0]
            if all(math.isfinite(value) for value in position_offset):
                arm["engine_model_source_position_offset_json"] = json.dumps(
                    list(position_offset), separators=(",", ":")
                )
            if math.isfinite(vertex_mpu) and vertex_mpu > 0.0:
                arm["engine_model_source_vertex_mpu"] = float(vertex_mpu)
                arm["engine_mpu"] = float(vertex_mpu)
    elif model_built:
        mb_offset, mb_size = model_built
        try:
            if mb_size >= MODEL_BUILT_VERTEX_MPU_OFFSET + 4:
                bsphere = struct.unpack_from("<4f", data, mb_offset + MODEL_BUILT_BSPHERE_OFFSET)
                aabb = struct.unpack_from("<3f", data, mb_offset + MODEL_BUILT_AABB_EXTENTS_OFFSET)
                common_mpu = struct.unpack_from("<f", data, mb_offset + MODEL_BUILT_COMMON_MPU_OFFSET)[0]
                vertex_mpu = struct.unpack_from("<f", data, mb_offset + MODEL_BUILT_VERTEX_MPU_OFFSET)[0]
                if (
                    all(math.isfinite(v) for v in bsphere + aabb)
                    and bsphere[3] > 0.0
                    and all(v >= 0.0 for v in aabb)
                ):
                    arm["engine_model_source_bsphere_json"] = json.dumps(list(bsphere), separators=(",", ":"))
                    arm["engine_model_source_aabb_json"] = json.dumps(list(aabb), separators=(",", ":"))
                if math.isfinite(common_mpu) and common_mpu > 0.0:
                    arm["engine_model_source_common_mpu"] = float(common_mpu)
                if math.isfinite(vertex_mpu) and vertex_mpu > 0.0:
                    arm["engine_model_source_vertex_mpu"] = float(vertex_mpu)
                    arm["engine_mpu"] = float(vertex_mpu)
                elif math.isfinite(common_mpu) and common_mpu > 0.0:
                    arm["engine_mpu"] = float(common_mpu)
        except Exception:
            pass


def _set_mesh_uv_layer(mesh, layer_name, vertex_indices, vertex_uvs):
    uv_layer = mesh.uv_layers.new(name=layer_name)
    if vertex_uvs is None or len(vertex_uvs) == 0:
        for item in uv_layer.data:
            item.uv = _engine_uv_to_blender((0.0, 0.0))
        return
    for loop_index, vertex_index in enumerate(vertex_indices):
        uv = vertex_uvs[int(vertex_index)]
        uv_layer.data[loop_index].uv = _engine_uv_to_blender(uv)


def _set_msmr_color_attributes(mesh, packed_colors, packed_words):
    """Expose MSMR's RGBA8 payload both as a color and linear mask channels."""
    normalized = np.asarray(packed_colors, dtype=np.float32) / np.float32(255.0)
    color_attribute = mesh.color_attributes.new(name="MSMR_Color", type='BYTE_COLOR', domain='POINT')
    color_attribute.data.foreach_set("color_srgb", normalized.reshape(-1))

    for channel_index, channel_name in enumerate(("R", "G", "B", "A")):
        mask_attribute = mesh.attributes.new(
            name=f"MSMR_Mask_{channel_name}",
            type='FLOAT',
            domain='POINT',
        )
        mask_attribute.data.foreach_set("value", normalized[:, channel_index])

    words = np.asarray(packed_words, dtype=np.uint32).astype(np.int64)
    signed_words = np.where(words < 0x80000000, words, words - 0x100000000)
    source_attribute = mesh.attributes.new(
        name="engine_source_vertex_color",
        type='INT',
        domain='POINT',
    )
    source_attribute.data.foreach_set("value", signed_words)


def _set_msmr_tangent_attributes(mesh, tangents, bitangents, bitangent_signs, tangent_valid):
    """Expose the decoded source tangent frame in Blender object coordinates."""
    blender_tangents = np.empty_like(tangents, dtype=np.float32)
    blender_tangents[:, 0] = tangents[:, 0]
    blender_tangents[:, 1] = -tangents[:, 2]
    blender_tangents[:, 2] = tangents[:, 1]
    tangent_attribute = mesh.attributes.new(
        name="MSMR_Tangent",
        type='FLOAT_VECTOR',
        domain='POINT',
    )
    tangent_attribute.data.foreach_set("vector", blender_tangents.reshape(-1))

    blender_bitangents = np.empty_like(bitangents, dtype=np.float32)
    blender_bitangents[:, 0] = bitangents[:, 0]
    blender_bitangents[:, 1] = -bitangents[:, 2]
    blender_bitangents[:, 2] = bitangents[:, 1]
    bitangent_attribute = mesh.attributes.new(
        name="MSMR_Bitangent",
        type='FLOAT_VECTOR',
        domain='POINT',
    )
    bitangent_attribute.data.foreach_set("vector", blender_bitangents.reshape(-1))

    sign_attribute = mesh.attributes.new(
        name="MSMR_BitangentSign",
        type='FLOAT',
        domain='POINT',
    )
    sign_attribute.data.foreach_set("value", bitangent_signs)
    valid_attribute = mesh.attributes.new(
        name="MSMR_TangentValid",
        type='BOOLEAN',
        domain='POINT',
    )
    valid_attribute.data.foreach_set("value", np.asarray(tangent_valid, dtype=np.bool_))


def _store_msmr_joint_metadata(arm, joints, metadata):
    """Attach lookup hashes and mirror partners to the imported skeleton."""
    lookup_entries = metadata.get("lookup_entries", []) if metadata else []
    mirror_records = metadata.get("mirror_records", []) if metadata else []
    arm["engine_model_joint_lookup_count"] = int(len(lookup_entries))
    arm["engine_model_joint_lookup_sentinel_present"] = bool(
        metadata and metadata.get("lookup_sentinel_present", False)
    )
    arm["engine_model_joint_lookup_json"] = json.dumps(
        lookup_entries, separators=(",", ":")
    )

    named_mirror_records = []
    for record in mirror_records:
        source_index = int(record["source_joint"])
        partner_index = int(record["partner_joint"])
        named_record = dict(record)
        named_record["source_name"] = str(joints[source_index]["name"])
        named_record["partner_name"] = str(joints[partner_index]["name"])
        named_mirror_records.append(named_record)
    arm["engine_model_mirror_record_count"] = int(len(mirror_records))
    arm["engine_model_mirror_pair_count"] = int(sum(
        int(record["source_joint"]) != int(record["partner_joint"])
        for record in mirror_records
    ))
    arm["engine_model_mirror_ids_json"] = json.dumps(
        named_mirror_records, separators=(",", ":")
    )

    lookup_by_joint = metadata.get("lookup_by_joint", []) if metadata else []
    mirror_by_joint = metadata.get("mirror_by_joint", []) if metadata else []
    for joint_index, joint in enumerate(joints):
        bone = arm.data.bones[joint["blender_name"]]
        if joint_index < len(lookup_by_joint) and lookup_by_joint[joint_index] is not None:
            bone["engine_joint_lookup_present"] = True
            bone["engine_joint_lookup_hash"] = to_signed_32(lookup_by_joint[joint_index])
        else:
            bone["engine_joint_lookup_present"] = False
        if joint_index >= len(mirror_by_joint) or mirror_by_joint[joint_index] is None:
            continue
        mapping = mirror_by_joint[joint_index]
        partner_index = int(mapping["partner_joint"])
        bone["engine_mirror_joint_index"] = partner_index
        bone["engine_mirror_joint_name"] = str(joints[partner_index]["name"])
        bone["engine_mirror_flags"] = int(mapping["flags"])
        bone["engine_mirror_record_index"] = int(mapping["record_index"])
        bone["engine_mirror_record_source"] = bool(mapping["record_source"])
        bone["engine_mirror_self"] = partner_index == joint_index


def _import_msmr_locators(context, data, blocks, arm, joints):
    """Represent game attachment/helper transforms as Blender Empty objects."""
    locators = decode_msmr_locators(data, blocks)
    arm["engine_model_locator_count"] = int(len(locators))
    if not locators:
        return []

    parent_collection = arm.users_collection[0] if arm.users_collection else context.collection
    locator_collection = bpy.data.collections.new(f"{arm.name}_Locators")
    parent_collection.children.link(locator_collection)
    locator_collection["engine_model_locator_collection"] = True
    locator_collection["engine_model_armature"] = arm.name
    arm["engine_model_locator_collection"] = locator_collection.name

    imported = []
    for locator in locators:
        locator_index = int(locator["index"])
        parent_joint = int(locator["parent_joint"])
        local_matrix = mathutils.Matrix(np.asarray(locator["matrix"], dtype=np.float64).tolist())
        empty = bpy.data.objects.new(str(locator["name"]), None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 0.035
        empty.show_in_front = True
        empty.hide_render = True
        locator_collection.objects.link(empty)

        empty["engine_model_locator"] = True
        empty["engine_locator_index"] = locator_index
        empty["engine_locator_name"] = str(locator["name"])
        empty["engine_locator_hash"] = to_signed_32(locator["hash"])
        empty["engine_locator_parent_joint"] = parent_joint
        empty["engine_locator_zero"] = int(locator["zero"])
        empty["engine_locator_matrix_json"] = json.dumps(
            locator["matrix_rows"], separators=(",", ":")
        )

        if 0 <= parent_joint < len(joints):
            bone_name = joints[parent_joint]["blender_name"]
            bone_matrix = arm.data.bones[bone_name].matrix_local
            world_matrix = arm.matrix_world @ bone_matrix @ local_matrix
            empty.parent = arm
            empty.parent_type = 'BONE'
            empty.parent_bone = bone_name
            empty.matrix_world = world_matrix
            empty["engine_locator_parent_bone"] = bone_name
        elif parent_joint == -1:
            world_matrix = arm.matrix_world @ SWIZZLE_MAT @ local_matrix
            empty.parent = arm
            empty.parent_type = 'OBJECT'
            empty.matrix_world = world_matrix
        else:
            raise ValueError(
                f"MSMR locator {locator.get('name', locator_index)} references joint {parent_joint} "
                f"outside the model skeleton"
            )
        imported.append(empty)
    return imported


def _set_curves_int_attribute(curve_data, name, domain, values):
    """Store a game integer stream without losing unsigned 32-bit bit patterns."""
    values = np.asarray(values)
    if values.dtype.itemsize >= 4 and values.dtype.kind == 'u':
        widened = values.astype(np.int64)
        values = np.where(widened < 0x80000000, widened, widened - 0x100000000)
    attribute = curve_data.attributes.new(name=name, type='INT', domain=domain)
    attribute.data.foreach_set("value", values.reshape(-1))


def _import_msmr_splines(context, data, blocks, arm):
    """Import Remastered hair/fur spline subsets as native poly Curves objects."""
    spline_model = decode_msmr_splines(data, blocks)
    arm["engine_model_spline_subset_count"] = 0
    arm["engine_model_spline_count"] = 0
    arm["engine_model_spline_point_count"] = 0
    if spline_model is None:
        return []

    arm["engine_model_spline_subset_count"] = int(len(spline_model["subsets"]))
    arm["engine_model_spline_count"] = int(spline_model["spline_count"])
    arm["engine_model_spline_point_count"] = int(spline_model["point_count"])
    arm["engine_model_spline_skin_binding_present"] = bool(
        spline_model["skin_bindings"] is not None
    )
    arm["engine_model_spline_skinning_decoded"] = False
    arm["engine_model_spline_joint_bindings_present"] = bool(
        spline_model["has_joint_bindings"]
    )
    arm["engine_model_spline_joint_weights_present"] = bool(
        spline_model["has_joint_weights"]
    )

    parent_collection = arm.users_collection[0] if arm.users_collection else context.collection
    spline_collection = bpy.data.collections.new(f"{arm.name}_Splines")
    parent_collection.children.link(spline_collection)
    spline_collection["engine_model_spline_collection"] = True
    spline_collection["engine_model_armature"] = arm.name
    arm["engine_model_spline_collection"] = spline_collection.name

    point_offsets = spline_model["point_offsets"]
    imported = []
    for subset in spline_model["subsets"]:
        first_spline = int(subset["first_spline"])
        spline_count = int(subset["spline_count"])
        last_spline = first_spline + spline_count
        first_point = int(point_offsets[first_spline])
        last_point = int(point_offsets[last_spline])
        point_count = last_point - first_point

        curve_data = bpy.data.hair_curves.new(str(subset["name"]))
        curve_data.add_curves(
            spline_model["point_counts"][first_spline:last_spline].astype(np.int32).tolist()
        )
        curve_data.set_types(type='POLY')

        positions = spline_model["positions"][first_point:last_point]
        blender_positions = np.empty(point_count * 3, dtype=np.float32)
        blender_positions[0::3] = positions[:, 0]
        blender_positions[1::3] = -positions[:, 2]
        blender_positions[2::3] = positions[:, 1]
        curve_data.points.foreach_set("position", blender_positions)

        _set_curves_int_attribute(
            curve_data,
            "engine_spline_point_w",
            'POINT',
            spline_model["point_w"][first_point:last_point],
        )
        _set_curves_int_attribute(
            curve_data,
            "engine_spline_curve_index",
            'CURVE',
            np.arange(first_spline, last_spline, dtype=np.int32),
        )
        for attribute_name, source_name in (
            ("engine_spline_value_u16", "value_u16"),
            ("engine_spline_lane", "lane"),
            ("engine_spline_packed_0", "packed_0"),
            ("engine_spline_packed_1", "packed_1"),
        ):
            _set_curves_int_attribute(
                curve_data,
                attribute_name,
                'CURVE',
                spline_model[source_name][first_spline:last_spline],
            )
        if spline_model["skin_bindings"] is not None:
            bindings = spline_model["skin_bindings"][first_spline:last_spline]
            for binding_index in range(4):
                _set_curves_int_attribute(
                    curve_data,
                    f"engine_spline_binding_{binding_index}",
                    'CURVE',
                    bindings[:, binding_index],
                )

        obj = bpy.data.objects.new(str(subset["name"]), curve_data)
        spline_collection.objects.link(obj)
        obj.parent = arm
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        obj.matrix_basis = mathutils.Matrix.Identity(4)

        obj["engine_model_spline_subset"] = True
        obj["engine_spline_subset_index"] = int(subset["index"])
        obj["engine_spline_subset_hash"] = to_signed_32(subset["hash"])
        obj["engine_spline_subset_name"] = str(subset["name"])
        obj["engine_spline_first_curve"] = first_spline
        obj["engine_spline_curve_count"] = spline_count
        obj["engine_spline_point_count"] = point_count
        obj["engine_spline_position_scale"] = float(spline_model["position_scale"])
        obj["engine_spline_unknown_floats_json"] = json.dumps(
            subset["unknown_floats"], separators=(",", ":")
        )
        obj["engine_spline_unknown_fields_json"] = json.dumps(
            subset["unknown_fields"], separators=(",", ":")
        )
        obj["engine_spline_string_paths_json"] = json.dumps(
            subset["string_paths"], separators=(",", ":")
        )
        obj["engine_spline_fur_tint_texture"] = str(subset["fur_tint_texture"])
        obj["engine_spline_fur_mask_texture"] = str(subset["fur_mask_texture"])
        obj["engine_spline_config_path"] = str(subset["config_path"])
        obj["engine_spline_skin_binding_present"] = bool(
            spline_model["skin_bindings"] is not None
        )
        obj["engine_spline_skinning_decoded"] = False
        curve_data["engine_model_spline_subset"] = True
        curve_data["engine_spline_subset_index"] = int(subset["index"])
        imported.append(obj)
    return imported


def _msmr_subset_lod_mask(looks, subset_index):
    mask = 0
    for look in looks or []:
        for lod_index, lod in enumerate(look.get("lods", [])):
            start = int(lod.get("start", 0))
            count = int(lod.get("count", 0))
            if count > 0 and start <= int(subset_index) < start + count:
                mask |= 1 << lod_index
    return mask


def _import_msmr_subsets(
    context,
    wm,
    data,
    blocks,
    arm,
    joints,
    imported_materials,
    import_all_lods,
    lod0_subset_ids,
):
    """Create Blender meshes from Remastered's shared geometry streams."""
    geometry = decode_msmr_geometry(data, blocks)
    subset_offset, subset_size = blocks[BLOCK_HASHES["ModelSubset"]]
    subset_count = subset_size // MSMR_SUBSET_RECORD_SIZE
    joint_count = len(joints)
    joint_names = [joint["blender_name"] for joint in joints]
    looks = _parse_model_looks_metadata(data, blocks)
    subset_objects = {}
    imported_count = 0

    for subset_index in range(subset_count):
        wm.progress_update(30 + int(65 * (subset_index / max(1, subset_count))))
        if not import_all_lods and lod0_subset_ids and subset_index not in lod0_subset_ids:
            continue

        subset = parse_msmr_subset(data, subset_offset, subset_index)
        vertex_start = subset["vertex_start"]
        vertex_count = subset["vertex_count"]
        index_count = subset["index_count"]
        if vertex_count <= 0 or index_count <= 0:
            continue
        if (
            vertex_start < 0
            or vertex_start > len(geometry["positions"])
            or vertex_count > len(geometry["positions"]) - vertex_start
        ):
            raise ValueError(f"MSMR subset {subset_index} references vertices outside Model Std Vert")

        index_values = msmr_subset_indices(geometry, subset)
        triangle_index_count = (len(index_values) // TRIANGLE_INDEX_COUNT) * TRIANGLE_INDEX_COUNT
        if triangle_index_count <= 0:
            continue
        index_values = index_values[:triangle_index_count]
        triangle_count = triangle_index_count // TRIANGLE_INDEX_COUNT

        positions = geometry["positions"][vertex_start:vertex_start + vertex_count]
        normals = geometry["normals"][vertex_start:vertex_start + vertex_count]
        uv0 = geometry["uv0"][vertex_start:vertex_start + vertex_count]
        uv1 = geometry["uv1"]
        if uv1 is not None:
            uv1 = uv1[vertex_start:vertex_start + vertex_count]
        colors = geometry["colors"]
        color_words = geometry["color_words"]
        if colors is not None:
            colors = colors[vertex_start:vertex_start + vertex_count]
            color_words = color_words[vertex_start:vertex_start + vertex_count]

        vertices_flat = np.empty(vertex_count * 3, dtype=np.float32)
        vertices_flat[0::3] = positions[:, 0]
        vertices_flat[1::3] = -positions[:, 2]
        vertices_flat[2::3] = positions[:, 1]

        mesh = bpy.data.meshes.new(f"Subset_{subset_index}")
        mesh.vertices.add(vertex_count)
        mesh.vertices.foreach_set("co", vertices_flat)
        mesh.loops.add(triangle_index_count)
        mesh.loops.foreach_set("vertex_index", index_values)
        mesh.polygons.add(triangle_count)
        mesh.polygons.foreach_set(
            "loop_start",
            np.arange(0, triangle_index_count, TRIANGLE_INDEX_COUNT, dtype=np.int32),
        )
        mesh.polygons.foreach_set(
            "loop_total",
            np.full(triangle_count, TRIANGLE_INDEX_COUNT, dtype=np.int32),
        )
        _set_mesh_uv_layer(mesh, "UV0", index_values, uv0)
        if uv1 is not None:
            _set_mesh_uv_layer(mesh, "UV1", index_values, uv1)
        mesh.update(calc_edges=True)
        mesh.validate(verbose=False)

        blender_normals = np.empty_like(normals)
        blender_normals[:, 0] = normals[:, 0]
        blender_normals[:, 1] = -normals[:, 2]
        blender_normals[:, 2] = normals[:, 1]
        mesh.normals_split_custom_set([
            blender_normals[int(mesh_loop.vertex_index)] for mesh_loop in mesh.loops
        ])

        position_ws = geometry["position_ws"][vertex_start:vertex_start + vertex_count]
        position_w_attr = mesh.attributes.new(name="engine_position_w", type='INT', domain='POINT')
        position_w_attr.data.foreach_set("value", position_ws)
        normal_words = geometry["normal_words"][vertex_start:vertex_start + vertex_count].astype(np.int64)
        signed_normal_words = np.where(normal_words < 0x80000000, normal_words, normal_words - 0x100000000)
        normal_attr = mesh.attributes.new(name="engine_source_normal_tangent", type='INT', domain='POINT')
        normal_attr.data.foreach_set("value", signed_normal_words)
        source_position_attr = mesh.attributes.new(
            name="engine_source_position", type='FLOAT_VECTOR', domain='POINT'
        )
        source_position_attr.data.foreach_set("vector", vertices_flat)
        source_vertex_index_attr = mesh.attributes.new(
            name="engine_source_vertex_index", type='INT', domain='POINT'
        )
        source_vertex_index_attr.data.foreach_set(
            "value",
            np.arange(vertex_count, dtype=np.int32),
        )
        source_uv0_u_attr = mesh.attributes.new(name="engine_source_uv0_u", type='FLOAT', domain='POINT')
        source_uv0_v_attr = mesh.attributes.new(name="engine_source_uv0_v", type='FLOAT', domain='POINT')
        source_uv0_u_attr.data.foreach_set("value", uv0[:, 0])
        source_uv0_v_attr.data.foreach_set("value", uv0[:, 1])
        if colors is not None:
            _set_msmr_color_attributes(mesh, colors, color_words)
        _set_msmr_tangent_attributes(
            mesh,
            geometry["tangents"][vertex_start:vertex_start + vertex_count],
            geometry["bitangents"][vertex_start:vertex_start + vertex_count],
            geometry["bitangent_signs"][vertex_start:vertex_start + vertex_count],
            geometry["tangent_valid"][vertex_start:vertex_start + vertex_count],
        )

        mesh.calc_loop_triangles()
        mesh["engine_source_topology_signature"] = model_topology_signature(
            int(mesh.loops[loop_index].vertex_index)
            for triangle in mesh.loop_triangles
            for loop_index in triangle.loops
        )
        mesh["engine_source_corner_normal_signature"] = model_corner_normal_signature(mesh.corner_normals)

        obj = bpy.data.objects.new(f"Subset_{subset_index}", mesh)
        context.collection.objects.link(obj)
        material_index = subset["material_index"]
        if 0 <= material_index < len(imported_materials):
            mesh.materials.append(imported_materials[material_index])
            for polygon in mesh.polygons:
                polygon.material_index = 0

        lod_mask = _msmr_subset_lod_mask(looks, subset_index)
        obj["engine_subset_index"] = int(subset_index)
        obj["engine_subset_flags"] = int(subset["flags"])
        obj["engine_material_index"] = int(material_index)
        obj["engine_uv_log_scales"] = 0
        obj["engine_mpu"] = float(geometry["position_scale"])
        obj["engine_lod_mask"] = int(lod_mask)
        obj["engine_uv0_present"] = True
        obj["engine_uv1_present"] = uv1 is not None
        obj["engine_uv2_present"] = False
        obj["engine_vertex_color_present"] = colors is not None
        obj["engine_tangent_frame_present"] = bool(
            geometry["tangent_valid"][vertex_start:vertex_start + vertex_count].any()
        )
        obj["engine_tangent_attribute"] = "MSMR_Tangent"
        obj["engine_bitangent_attribute"] = "MSMR_Bitangent"
        if colors is not None:
            obj["engine_vertex_color_attribute"] = "MSMR_Color"
        obj.parent = arm
        subset_objects[subset_index] = obj

        if joint_count > 0:
            modifier = obj.modifiers.new(type='ARMATURE', name="Armature")
            modifier.object = arm
            vertex_groups = [obj.vertex_groups.new(name=name) for name in joint_names]
            if subset["flags"] & SUBSET_FLAG_SKINNED:
                grouped_weights = decode_msmr_skin_weights(data, blocks, subset, joint_count)
                for joint_index, weights in grouped_weights.items():
                    vertex_group = vertex_groups[joint_index]
                    for weight, vertex_indices in weights.items():
                        vertex_group.add(vertex_indices, float(weight) / SKIN_WEIGHT_SCALE, 'REPLACE')

        imported_count += 1

    arm["engine_model_imported_subset_count"] = int(imported_count)
    return subset_objects


class ImportEngineModel(Operator, ImportHelper):
    bl_idname = "import_scene.engine_model"
    bl_label = "Import Luna Engine Model"
    bl_description = "Import one or more engine-compiled binary Model (.model) files"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    import_all_lods: BoolProperty(
        name="Import All LODs",
        description="Import lower-detail subsets too; off imports only LOD0 subsets",
        default=False,
        options={'SKIP_SAVE'},
    )
    import_shape_keys: BoolProperty(
        name="Import Shape Keys",
        description="Import model morph targets and bake named Ziva channels into ordinary Blender shape keys",
        default=True,
        options={'SKIP_SAVE'},
    )

    def draw(self, context):
        self.layout.prop(self, "import_all_lods")
        self.layout.prop(self, "import_shape_keys")

    def _resolve_filepaths(self):
        paths = []
        selected_files = list(getattr(self, "files", []) or [])
        if selected_files:
            base = self.directory or os.path.dirname(self.filepath or "")
            for entry in selected_files:
                name = getattr(entry, "name", "") or ""
                if not name:
                    continue
                paths.append(os.path.join(base, name))
        if not paths and self.filepath:
            paths.append(self.filepath)

        unique_paths = []
        seen = set()
        for path in paths:
            if not path:
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)
        return unique_paths

    def execute(self, context):
        filepaths = self._resolve_filepaths()
        if not filepaths:
            self.report({'ERROR'}, "Choose at least one .model file, then click Import again.")
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        success_count = 0
        failure_messages = []
        reported_failure_count = 0
        try:
            total = len(filepaths)
            for index, filepath in enumerate(filepaths):
                if not os.path.isfile(filepath):
                    failure_messages.append(
                        f"{os.path.basename(filepath) or filepath}: the file was moved or deleted; choose it again"
                    )
                    continue
                try:
                    result = self._do_import(context, wm, filepath)
                except Exception as exc:
                    log_exception("Unexpected error importing model %s", filepath)
                    failure_messages.append(
                        f"{os.path.basename(filepath)}: couldn't be read; make sure it is an original "
                        "extracted game model"
                    )
                    continue
                if result == {'FINISHED'}:
                    success_count += 1
                else:
                    reported_failure_count += 1
                    failure_messages.append(
                        f"{os.path.basename(filepath)}: couldn't be imported; follow the message shown above"
                    )
                if total > 1:
                    wm.progress_update(int(100 * ((index + 1) / total)))
        finally:
            wm.progress_end()

        if success_count == 0:
            if reported_failure_count == len(failure_messages):
                return {'CANCELLED'}
            detail = "; ".join(failure_messages) if failure_messages else "no files imported"
            self.report(
                {'ERROR'},
                f"I couldn't import this model. {detail}. Choose an original game .model file and try again.",
            )
            return {'CANCELLED'}

        if failure_messages:
            self.report(
                {'WARNING'},
                f"Imported {success_count} of {len(filepaths)} models. These files still need attention: "
                f"{'; '.join(failure_messages)}. Choose those files again after fixing the issue.",
            )
        else:
            self.report({'INFO'}, f"Imported {success_count} model{'s' if success_count != 1 else ''}.")
        return {'FINISHED'}

    def _do_import(self, context, wm, filepath):
        data, blocks, sb_offset = get_dat1_data(filepath)
        if not data:
            self.report(
                {'ERROR'},
                f"{os.path.basename(filepath)} isn't a supported game model. Choose the original extracted "
                ".model file, not a renamed or converted file.",
            )
            return {'CANCELLED'}
        msmr = is_msmr_model(blocks)
        required_model_blocks = (
            "ModelBuilt",
            "ModelMaterial",
            "ModelLook",
            "ModelLookBuilt",
            "ModelLookGroup",
            "ModelSubset",
        )
        if msmr:
            required_model_blocks += ("ModelIndex", "ModelStdVert")
        else:
            required_model_blocks += ("ModelSubsetGeomData",)
        missing_model_blocks = [name for name in required_model_blocks if BLOCK_HASHES[name] not in blocks]
        if missing_model_blocks:
            self.report(
                {'ERROR'},
                f"{os.path.basename(filepath)} doesn't contain a complete game model. Choose a different "
                f"original .model file. Missing internal data: {', '.join(missing_model_blocks)}.",
            )
            return {'CANCELLED'}

        wm.progress_update(5)
        import_all_lods = bool(getattr(self, "import_all_lods", False))
        import_shape_keys = bool(getattr(self, "import_shape_keys", True))
        has_morphs = has_msmr_morphs(blocks) if msmr else BLOCK_HASHES.get("ModelAnimMorph2Info") in blocks
        has_ziva = BLOCK_HASHES.get("ModelAnimZiva2Info") in blocks
        ziva_model = load_ziva_model(filepath) if has_ziva else None
        lod0_subset_ids = _collect_model_lod_subset_ids(_parse_model_looks_metadata(data, blocks), 0)
        morph = None
        if import_shape_keys and has_morphs:
            if msmr:
                morph = decode_msmr_morphs(
                    data,
                    blocks,
                    allowed_subset_ids=None if import_all_lods or not lod0_subset_ids else lod0_subset_ids,
                )
            else:
                morph = decode_model_morph2(data, blocks)
        has_skeleton = all(
            BLOCK_HASHES[name] in blocks
            for name in ("ModelJointHierarchy", "ModelJoint", "ModelBindPose")
        )

        dat1_version = struct.unpack_from("<I", data, 4)[0]
        source_had_stg = False
        try:
            with open(filepath, "rb") as src:
                source_had_stg = src.read(4) == b"STG\x00"
        except OSError:
            pass

        # MODEL assets are expected to carry the 48-byte STG wrapper. Reset
        # this model-export setting on every import so an older .blend or an
        # already-open scene cannot silently keep exporting raw DAT1.
        context.scene.engine_export_add_stg_header = True

        mpu = 1.0
        if msmr:
            built_off, built_size = blocks[BLOCK_HASHES["ModelBuilt"]]
            if built_size >= 48:
                candidate_mpu = struct.unpack_from("<f", data, built_off + 44)[0]
                if math.isfinite(candidate_mpu) and candidate_mpu > 0.0:
                    mpu = candidate_mpu
        elif BLOCK_HASHES["ModelSubset"] in blocks:
            s_off, _ = blocks[BLOCK_HASHES["ModelSubset"]]
            mpu = struct.unpack_from("<f", data, s_off + MODEL_SUBSET_MPU_OFFSET)[0]
        material_infos = _parse_model_materials(data, blocks)
        imported_materials = [_make_import_material(material_info) for material_info in material_infos]

        joint_count = 0
        joints = []
        if has_skeleton:
            h_off, _ = blocks[BLOCK_HASHES["ModelJointHierarchy"]]
            joint_count = struct.unpack_from("<HHHH", data, h_off)[1]
            j_off, _ = blocks[BLOCK_HASHES["ModelJoint"]]
            b_off, _ = blocks[BLOCK_HASHES["ModelBindPose"]]

            bp_raw = struct.unpack_from(f"<{MODEL_BIND_POSE_FLOATS_PER_JOINT * joint_count}f", data, b_off)
            local_bind_mats = []
            for i in range(joint_count):
                base = i * MODEL_BIND_POSE_FLOATS_PER_JOINT
                loc = mathutils.Matrix.Translation((bp_raw[base+8], bp_raw[base+9], bp_raw[base+10]))
                rot = mathutils.Quaternion((bp_raw[base+7], bp_raw[base+4], bp_raw[base+5], bp_raw[base+6])).to_matrix().to_4x4()
                sca = mathutils.Matrix.Diagonal((bp_raw[base], bp_raw[base+1], bp_raw[base+2], 1.0))
                local_bind_mats.append(loc @ rot @ sca)

            j_raws = list(struct.iter_unpack("<hHHHII", data[j_off:j_off + joint_count * MODEL_JOINT_RECORD_SIZE]))
            for i in range(joint_count):
                j_raw = j_raws[i]
                if msmr:
                    # Remastered stores an absolute DAT1 string offset.  The
                    # newer layout stores a relative offset instead.
                    name = _read_c_string(data, j_raw[5], MODEL_JOINT_NAME_BYTES)
                else:
                    name_addr = j_off + i * MODEL_JOINT_RECORD_SIZE + j_raw[5]
                    name = data[name_addr:name_addr + MODEL_JOINT_NAME_BYTES].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                    if not name or not all(c.isprintable() for c in name):
                        sb_name_addr = sb_offset + j_raw[5]
                        name = data[sb_name_addr:sb_name_addr + MODEL_JOINT_NAME_BYTES].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                if not name:
                    name = f"Joint_{j_raw[4]:08X}"
                joints.append({
                    'parent': int(j_raw[0]),
                    'name': name,
                    'local_mat': local_bind_mats[i],
                    'source_index': int(j_raw[1]),
                    'unknown_1': int(j_raw[2]),
                    'unknown_2': int(j_raw[3]),
                    'hash': int(j_raw[4]) & U32_MASK,
                })

        joint_metadata = None
        if msmr and has_skeleton:
            joint_metadata = decode_msmr_joint_metadata(
                data,
                blocks,
                joint_count,
                [joint["hash"] for joint in joints],
            )

        wm.progress_update(15)


        bpy.ops.object.armature_add(enter_editmode=True)
        arm = context.active_object
        arm.name = os.path.basename(filepath)
        while arm.data.edit_bones:
            arm.data.edit_bones.remove(arm.data.edit_bones[0])

        global_mats = [None] * joint_count
        if has_skeleton:
            for i in range(joint_count):
                eb = arm.data.edit_bones.new(joints[i]['name'])
                joints[i]['blender_name'] = eb.name

            for i, j in enumerate(joints):
                if j['parent'] != -1:
                    arm.data.edit_bones[j['blender_name']].parent = arm.data.edit_bones[joints[j['parent']]['blender_name']]

            for i in range(joint_count):
                j = joints[i]
                eb = arm.data.edit_bones[j['blender_name']]
                if j['parent'] == -1:
                    global_mats[i] = SWIZZLE_MAT @ j['local_mat']
                else:
                    global_mats[i] = global_mats[j['parent']] @ j['local_mat']
                eb.matrix = global_mats[i]
                eb.tail = eb.head + (global_mats[i].to_3x3() @ mathutils.Vector((0, 0.1, 0)))

        bpy.ops.object.mode_set(mode='OBJECT')
        for i, joint in enumerate(joints):
            bone = arm.data.bones[joint['blender_name']]
            bone["engine_joint_index"] = i
            bone["engine_joint_name"] = joint['name']
            bone["engine_joint_source_index"] = int(joint["source_index"])
            bone["engine_joint_hash"] = to_signed_32(joint["hash"])
            bone["engine_joint_unknown_1"] = int(joint["unknown_1"])
            bone["engine_joint_unknown_2"] = int(joint["unknown_2"])
        arm["engine_mpu"] = mpu
        arm["engine_model_static"] = not has_skeleton
        _store_model_metadata(
            arm,
            filepath,
            data,
            blocks,
            sb_offset,
            source_had_stg,
            dat1_version,
            joint_count,
            len(imported_materials),
        )
        if msmr and joint_metadata is not None:
            _store_msmr_joint_metadata(arm, joints, joint_metadata)
        else:
            arm["engine_model_joint_lookup_count"] = 0
            arm["engine_model_mirror_record_count"] = 0
            arm["engine_model_mirror_pair_count"] = 0
        arm["engine_model_import_all_lods"] = bool(import_all_lods)
        arm["engine_model_import_mode"] = "ALL_LODS" if import_all_lods else "LOD0"
        arm["engine_model_imported_subset_count"] = 0
        arm["engine_model_locator_count"] = 0
        arm["engine_model_spline_subset_count"] = 0
        arm["engine_model_spline_count"] = 0
        arm["engine_model_spline_point_count"] = 0
        arm["engine_model_source_has_morphs"] = bool(has_morphs)
        arm["engine_model_source_has_ziva"] = bool(has_ziva)
        arm["engine_model_shape_keys_imported"] = False
        arm["engine_model_ziva_shape_keys_imported"] = False
        arm["engine_model_imported_ziva_key_count"] = 0
        arm["engine_model_morph_search"] = ""
        arm["engine_ziva_mode"] = "SOURCE_ZIVA" if has_ziva else "NONE"
        arm["engine_ziva_metadata_json"] = json.dumps(
            ziva_model.metadata() if ziva_model is not None else {},
            separators=(",", ":"),
            sort_keys=True,
        )

        wm.progress_update(30)

        subset_objects = {}
        if msmr:
            subset_objects = _import_msmr_subsets(
                context,
                wm,
                data,
                blocks,
                arm,
                joints,
                imported_materials,
                import_all_lods,
                lod0_subset_ids,
            )
        elif BLOCK_HASHES["ModelSubset"] in blocks and BLOCK_HASHES["ModelSubsetGeomData"] in blocks:
            s_off, s_size = blocks[BLOCK_HASHES["ModelSubset"]]
            g_off, _ = blocks[BLOCK_HASHES["ModelSubsetGeomData"]]
            subset_count = s_size // MODEL_SUBSET_RECORD_SIZE
            joint_names_list = [j['blender_name'] for j in joints]
            imported_subset_count = 0

            for s in range(subset_count):
                wm.progress_update(30 + int(65 * (s / max(1, subset_count))))
                sub_ptr = s_off + s * MODEL_SUBSET_RECORD_SIZE
                idx_count = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_INDEX_COUNT_OFFSET)[0]
                vtx_count = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_COUNT_OFFSET)[0]
                idx_data_off = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_INDEX_DATA_OFFSET)[0]
                subset_flags = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_FLAGS_OFFSET)[0]
                subset_mpu = struct.unpack_from("<f", data, sub_ptr + MODEL_SUBSET_MPU_OFFSET)[0]
                if not math.isfinite(subset_mpu) or subset_mpu <= 0.0:
                    subset_mpu = mpu
                uv_log_scales = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_UV_LOG_OFFSET)[0]
                material_index = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_MATERIAL_INDEX_OFFSET)[0]
                v_std = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_STD_OFFSET)[0]
                v_uv12 = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_UV12_OFFSET)[0]
                s_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_SKIN_OFFSET)[0]
                c_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_CLUSTER_OFFSET)[0]
                b_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_BASE_OFFSET)[0]
                lod_mask = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_LOD_MASK_OFFSET)[0]
                if not import_all_lods:
                    is_lod0 = bool(lod_mask & 1)
                    if lod0_subset_ids:
                        is_lod0 = is_lod0 or (s in lod0_subset_ids)
                    elif lod_mask == 0:
                        is_lod0 = True
                    if not is_lod0:
                        continue

                if vtx_count == 0 or idx_count == 0:
                    continue

                v_base = g_off + b_off_v + v_std
                i_base = g_off + b_off_v + idx_data_off



                vtx_block = np.frombuffer(data, dtype=np.int16, count=vtx_count * MODEL_VERTEX_SHORTS_PER_VERTEX, offset=v_base).reshape(-1, MODEL_VERTEX_SHORTS_PER_VERTEX)
                xyz = vtx_block[:, :3].astype(np.float32) * subset_mpu
                if subset_flags & SUBSET_FLAG_ORIGIN_OFFSET:
                    xyz += _decode_subset_origin_meters(data, sub_ptr, subset_mpu)

                uv0 = vtx_block[:, 6:8].astype(np.float32) * _uv_fixed_to_float(uv_log_scales, 0)
                uv1 = None
                uv2 = None
                uv12_channel_count = int(bool(subset_flags & SUBSET_FLAG_HAS_UV1)) + int(bool(subset_flags & SUBSET_FLAG_HAS_UV2))
                if v_uv12 > 0 and uv12_channel_count:
                    uv12_base = g_off + b_off_v + v_uv12
                    uv12_values = np.frombuffer(
                        data,
                        dtype=np.int16,
                        count=vtx_count * uv12_channel_count * 2,
                        offset=uv12_base,
                    ).reshape(vtx_count, uv12_channel_count, 2).astype(np.float32)
                    if subset_flags & SUBSET_FLAG_HAS_UV1:
                        uv1 = uv12_values[:, 0, :] * _uv_fixed_to_float(uv_log_scales, 1)
                    if subset_flags & SUBSET_FLAG_HAS_UV2:
                        uv2_index = 1 if (subset_flags & SUBSET_FLAG_HAS_UV1) else 0
                        uv2 = uv12_values[:, uv2_index, :] * _uv_fixed_to_float(uv_log_scales, 2)

                verts_flat = np.empty(vtx_count * 3, dtype=np.float32)
                verts_flat[0::3] = xyz[:, 0]
                verts_flat[1::3] = -xyz[:, 2]
                verts_flat[2::3] = xyz[:, 1]


                idx_arr = np.frombuffer(data, dtype=np.uint16, count=idx_count, offset=i_base).astype(np.int32)
                tri_count = idx_count // TRIANGLE_INDEX_COUNT


                me = bpy.data.meshes.new(f"Subset_{s}")
                me.vertices.add(vtx_count)
                me.vertices.foreach_set("co", verts_flat)
                me.loops.add(tri_count * TRIANGLE_INDEX_COUNT)
                me.loops.foreach_set("vertex_index", idx_arr[:tri_count * TRIANGLE_INDEX_COUNT])
                me.polygons.add(tri_count)
                loop_starts = np.arange(0, tri_count * TRIANGLE_INDEX_COUNT, TRIANGLE_INDEX_COUNT, dtype=np.int32)
                loop_totals = np.full(tri_count, TRIANGLE_INDEX_COUNT, dtype=np.int32)
                me.polygons.foreach_set("loop_start", loop_starts)
                me.polygons.foreach_set("loop_total", loop_totals)
                loop_vertex_indices = idx_arr[:tri_count * TRIANGLE_INDEX_COUNT]
                _set_mesh_uv_layer(me, "UV0", loop_vertex_indices, uv0)
                _set_mesh_uv_layer(me, "UV1", loop_vertex_indices, uv1)
                _set_mesh_uv_layer(me, "UV2", loop_vertex_indices, uv2)
                me.update(calc_edges=True)
                me.validate(verbose=False)

                vertex_normals = []
                normal_tangents = []
                position_ws = []
                for vertex_index in range(vtx_count):
                    raw_vertex = struct.unpack_from("<hhhhIhh", data, v_base + vertex_index * 16)
                    engine_normal = _decode_packed_normal(raw_vertex[4])
                    vertex_normals.append((engine_normal[0], -engine_normal[2], engine_normal[1]))
                    normal_tangent = int(raw_vertex[4]) & U32_MASK
                    normal_tangents.append(
                        normal_tangent if normal_tangent < 0x80000000 else normal_tangent - 0x100000000
                    )
                    position_ws.append(int(raw_vertex[3]))
                me.normals_split_custom_set([
                    vertex_normals[int(mesh_loop.vertex_index)] for mesh_loop in me.loops
                ])
                position_w_attr = me.attributes.new(name="engine_position_w", type='INT', domain='POINT')
                position_w_attr.data.foreach_set("value", position_ws)
                normal_tangent_attr = me.attributes.new(
                    name="engine_source_normal_tangent",
                    type='INT',
                    domain='POINT',
                )
                normal_tangent_attr.data.foreach_set("value", normal_tangents)
                source_position_attr = me.attributes.new(
                    name="engine_source_position",
                    type='FLOAT_VECTOR',
                    domain='POINT',
                )
                source_position_attr.data.foreach_set("vector", verts_flat)
                source_uv0_u_attr = me.attributes.new(name="engine_source_uv0_u", type='FLOAT', domain='POINT')
                source_uv0_v_attr = me.attributes.new(name="engine_source_uv0_v", type='FLOAT', domain='POINT')
                source_uv0_u_attr.data.foreach_set("value", uv0[:, 0])
                source_uv0_v_attr.data.foreach_set("value", uv0[:, 1])
                me.calc_loop_triangles()
                me["engine_source_topology_signature"] = model_topology_signature(
                    int(me.loops[loop_index].vertex_index)
                    for triangle in me.loop_triangles
                    for loop_index in triangle.loops
                )
                me["engine_source_corner_normal_signature"] = model_corner_normal_signature(
                    me.corner_normals
                )

                obj = bpy.data.objects.new(f"Subset_{s}", me)
                context.collection.objects.link(obj)
                if 0 <= material_index < len(imported_materials):
                    me.materials.append(imported_materials[material_index])
                    for poly in me.polygons:
                        poly.material_index = 0
                obj["engine_subset_index"] = s
                obj["engine_subset_flags"] = int(subset_flags)
                obj["engine_material_index"] = int(material_index)
                obj["engine_uv_log_scales"] = int(uv_log_scales)
                obj["engine_mpu"] = float(subset_mpu)
                obj["engine_lod_mask"] = lod_mask
                obj["engine_uv0_present"] = True
                obj["engine_uv1_present"] = bool(subset_flags & SUBSET_FLAG_HAS_UV1)
                obj["engine_uv2_present"] = bool(subset_flags & SUBSET_FLAG_HAS_UV2)
                if ziva_model is not None and s < len(ziva_model.subsets):
                    ziva_subset = ziva_model.subsets[s]
                    obj["engine_ziva_source_mapped"] = ziva_subset["elem_index"] != ZIVA_INVALID_ELEM
                    if ziva_subset["elem_index"] != ZIVA_INVALID_ELEM:
                        obj["engine_ziva_element_index"] = int(ziva_subset["elem_index"])
                subset_objects[int(s)] = obj
                obj.parent = arm
                if has_skeleton:
                    mod = obj.modifiers.new(type='ARMATURE', name="Armature")
                    mod.object = arm


                if has_skeleton and joint_count > 0 and s_off_v > 0 and c_off_v > 0:
                    vgs = [obj.vertex_groups.new(name=n) for n in joint_names_list]
                    c_b = g_off + b_off_v + c_off_v
                    s_b = g_off + b_off_v + s_off_v



                    group_weights = {}

                    cluster_count = (vtx_count + SKIN_CLUSTER_VERTEX_COUNT - 1) // SKIN_CLUSTER_VERTEX_COUNT

                    cluster_hdrs = struct.unpack_from(f"<{cluster_count}I", data, c_b)

                    for ic in range(cluster_count):
                        cl = cluster_hdrs[ic]
                        c_o = cl & SKIN_CLUSTER_OFFSET_MASK
                        c_inf = ((cl >> SKIN_CLUSTER_INFLUENCE_SHIFT) & SKIN_CLUSTER_NIBBLE_MASK) + 1
                        c_jo = (cl >> SKIN_CLUSTER_JOINT_OFFSET_SHIFT) & SKIN_CLUSTER_NIBBLE_MASK
                        cl_ptr = s_b + c_o * SKIN_CLUSTER_WORD_BYTES
                        is_full_idx = bool(cl & SKIN_CLUSTER_FULL_INDEX_BIT)
                        bpj = (2 if is_full_idx else 1) + (1 if c_inf > 1 else 0)

                        base_v = ic * SKIN_CLUSTER_VERTEX_COUNT
                        remaining = min(SKIN_CLUSTER_VERTEX_COUNT, vtx_count - base_v)

                        for vic in range(remaining):
                            v_idx = base_v + vic
                            v_ptr = cl_ptr + vic * (c_inf * bpj)
                            cur_ji = int(c_jo * SKIN_WEIGHT_SCALE)
                            for infl in range(c_inf):
                                jp = v_ptr + infl * bpj
                                if is_full_idx:
                                    ji = struct.unpack_from("<H", data, jp)[0]
                                    w = data[jp + 2] if c_inf > 1 else int(SKIN_WEIGHT_SCALE)
                                else:
                                    cur_ji += data[jp]
                                    ji = cur_ji
                                    w = data[jp + 1] if c_inf > 1 else int(SKIN_WEIGHT_SCALE)
                                if w > 0 and ji < joint_count:
                                    gw = group_weights.setdefault(ji, {})
                                    gw.setdefault(w, []).append(v_idx)


                    for ji, wdict in group_weights.items():
                        vg = vgs[ji]
                        for w, vlist in wdict.items():
                            vg.add(vlist, w / SKIN_WEIGHT_SCALE, 'REPLACE')
                imported_subset_count += 1
            arm["engine_model_imported_subset_count"] = int(imported_subset_count)

        imported_morph_keys = 0
        if morph is not None:
            imported_morph_keys = _import_morph_shape_keys(morph, subset_objects, arm, wm=wm)
        imported_ziva_keys = 0
        if import_shape_keys and ziva_model is not None and ziva_model.sliders:
            ziva_objects = [
                obj for obj in subset_objects.values()
                if bool(obj.get("engine_ziva_source_mapped", False))
            ]
            if ziva_objects:
                wm.progress_update(95)
                ziva_result = transfer_ziva_channels_to_objects(
                    ziva_model,
                    arm,
                    ziva_objects,
                    max_distance=0.001,
                    lod=None if import_all_lods else 0,
                    progress=lambda current, total, channel: wm.progress_update(
                        95 + int(4 * (current / max(1, total)))
                    ),
                )
                imported_ziva_keys = int(ziva_result["created"])
                arm["engine_ziva_mode"] = "SOURCE_ZIVA"
                arm["engine_model_shape_keys_imported"] = True
                arm["engine_model_ziva_shape_keys_imported"] = True
                arm["engine_model_morph_target_count"] = int(len(ziva_model.sliders))
        if imported_morph_keys or imported_ziva_keys:
            sync_armature_morph_controls(arm)
        arm["engine_model_imported_morph_key_count"] = int(imported_morph_keys)
        arm["engine_model_imported_ziva_key_count"] = int(imported_ziva_keys)

        imported_locators = _import_msmr_locators(context, data, blocks, arm, joints) if msmr else []
        imported_splines = _import_msmr_splines(context, data, blocks, arm) if msmr else []

        arm.active_lod = 0
        update_lod_visibility(arm, context)
        wm.progress_update(100)
        mode_label = "mesh parts from all detail levels" if import_all_lods else "main mesh parts"
        imported_count = int(arm.get("engine_model_imported_subset_count", 0))
        extras = []
        if import_shape_keys and has_morphs:
            extras.append(
                f"{int(arm.get('engine_model_imported_morph_key_count', 0))} morph shape-key copies"
            )
        elif import_shape_keys and has_ziva and ziva_model is not None and ziva_model.sliders:
            extras.append(
                f"{int(arm.get('engine_model_imported_ziva_key_count', 0))} facial shape-key copies "
                f"using {len(ziva_model.sliders)} controls"
            )
        if imported_locators:
            extras.append(f"{len(imported_locators)} model locators")
        if imported_splines:
            extras.append(f"{len(imported_splines)} spline subsets")
        extras_label = f", plus {' and '.join(extras)}" if extras else ""
        self.report(
            {'INFO'},
            f"Import finished: {joint_count} bones and {imported_count} "
            f"{mode_label}{extras_label}.",
        )
        return {'FINISHED'}
