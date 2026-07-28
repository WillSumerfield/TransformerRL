# vlearn 0.3.8 — Geometry & Material API Reference

## Overview

These APIs control runtime morphology, physics properties, and material friction/restitution of rigid bodies and articulations. All follow the standard vlearn three-phase command pattern:

```python
# 1. Allocate (setup, after gym.finalize())
buf = torch.zeros(SIZE, dtype=DTYPE, device="cuda:0")
cmd = env_group.create_XYZ_command(v.wrap_gpu_buffer(buf), handle, ...)
arr = gym.create_gpu_array([cmd])

# 2. Read physics → buffer
gym.get_XYZ(arr)

# 3. Write buffer → physics
gym.set_XYZ(arr)
```

### Critical ordering constraint

```
import_definitions()
  → create_rigid_body() / create_articulation()
  → env_def.finalize()
  → create_environment_group()
  → tile_environments()         # must be BEFORE gym.finalize()
  → gym.finalize()              # geometry handles become valid here
  → get_*_geometry_handle()    # only valid AFTER gym.finalize()
```

vlearn only allows **one `Gym` per process**. All env_groups must be created and positioned before `gym.finalize()`.

---

## 1. Rigid Body Morph Geometry

Reads and writes the deformation cage (control point mesh) of a morph-mesh rigid body. Moving control points deforms the collision shape.

### Required vsim XML

The collision link must use `morph_mesh` geometry:

```xml
<link name="box">
    <collision name="Collision 0">
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
            <morph_mesh filename="package://MORPHS/cube.obj" scale="10 10 10"/>
            <settings maxResPerAxis="128" targetSamples="1024"
                      sparseDistanceErrorThreshold="9.99999974738e-05"
                      voxelSize="0.10000000149"/>
            <queryProperties useQueries="false"/>
        </geometry>
    </collision>
</link>
```

### Setup

```python
# After gym.finalize():
rb_def_handle = env_def.get_rigid_body_def_handle_by_name("large_box")
geom_handle   = env_def.get_rigid_body_def_geometry_handle(rb_def_handle)
# geom_handle.index()  → int (0x03FFFFFF = invalid; otherwise valid)
# geom_handle.type()   → int (type code 10 = morph mesh)

num_verts = env_def.get_morph_geometry_num_vertices(geom_handle)
# Returns number of control cage vertices (e.g. 512 for 8×8×8 cage)
```

### Buffer format

```
cage_buf:  (num_verts * 3,)  float32 — flat [v0x, v0y, v0z, v1x, v1y, v1z, ...]
vert_weight_buf: (num_verts,) float32 — blend weight per vertex (1.0 = full effect)
```

**Cage buffer is NOT per-env** — it's a shared definition-level buffer. One cage applies to all environments.

### Usage

```python
import torch
import vlearn as v

N    = 4              # number of envs
DEV  = torch.device("cuda:0")

# (after gym.finalize)
geom_h    = env_def.get_rigid_body_def_geometry_handle(rb_def_h)
num_verts = env_def.get_morph_geometry_num_vertices(geom_h)

cage_buf = torch.zeros(num_verts * 3, dtype=torch.float32, device=DEV)
vw_buf   = torch.ones(num_verts,      dtype=torch.float32, device=DEV)

cmd = env_group.create_rigid_body_morph_geometry_command(
    v.wrap_gpu_buffer(cage_buf),   # deformation cage positions
    v.wrap_gpu_buffer(vw_buf),     # vertex weights
    geom_h,
    rb_def_h,
)
arr = gym.create_gpu_array([cmd])

# Read current cage from sim
gym.get_rigid_body_morph_geometry_deformation_cages(arr)
# cage_buf now has [v0x, v0y, v0z, v1x, ...] in local frame

# Modify: collapse cage to its centroid (squishes shape to a point)
verts    = cage_buf.reshape(num_verts, 3)
centroid = verts.mean(dim=0, keepdim=True).expand_as(verts).contiguous()
cage_buf[:] = centroid.reshape(-1)

# Write modified cage to sim
gym.set_rigid_body_morph_geometry_deformation_cages(arr, apply_smoothing=False)
# apply_smoothing=True runs a Laplacian smooth over the cage before applying
```

### Observed values (morph_cube.vsim, scale=10×10×10)

- `num_verts = 512` (8×8×8 grid)
- Initial cage range: `[-0.35, 0.35]` in all axes (local frame, before world transform)
- `geom_handle.index() = 0`, `type = 10`

---

## 2. Rigid Body Property Command

Reads and writes scalar physics properties (mass, inertia, body-to-model transform) of a rigid body definition.

### Buffer format

Buffers are **definition-level** (NOT per-env):

| Property            | Buffer size | dtype   | Notes |
|---------------------|-------------|---------|-------|
| `INV_MASS`          | 1 float     | float32 | inverse mass = 1/mass |
| `DIAG_INV_INERTIA`  | 3 floats    | float32 | diagonal of inverse inertia tensor |
| `BODY_TO_MODEL`     | 7 floats    | float32 | `[qx, qy, qz, qw, px, py, pz]` — quat + pos |

### Usage

```python
# Allocate once
inv_mass_buf   = torch.zeros(1, dtype=torch.float32, device=DEV)
diag_inv_buf   = torch.zeros(3, dtype=torch.float32, device=DEV)
body2model_buf = torch.zeros(7, dtype=torch.float32, device=DEV)

cmd_m = env_group.create_rigid_body_property_command(
    v.RigidBodyProperty.INV_MASS,
    v.wrap_gpu_buffer(inv_mass_buf),
    rb_def_handle,
)
arr_m = gym.create_gpu_array([cmd_m])

# SET: write new value to physics
inv_mass_buf[0] = 1.0 / 5.0          # 5 kg body
gym.set_rigid_body_properties(arr_m)

# GET: read current value back into buffer
gym.get_rigid_body_properties(arr_m)
print(inv_mass_buf[0])  # 0.2
```

**Important:** `get_rigid_body_properties` reads from the GPU command buffer, which is only populated after a `set` call. Initial GET before any SET may return uninitialized values. To check the def value at load time, read `env_def.get_rigid_body_def(rb_def_handle).inv_mass` instead.

### RigidBodyProperty enum values

```python
v.RigidBodyProperty.INV_MASS          # = 0
v.RigidBodyProperty.DIAG_INV_INERTIA  # = 1
v.RigidBodyProperty.BODY_TO_MODEL     # = 2
```

---

## 3. Rigid Material Property Command

Reads and writes friction/restitution/contact properties of a rigid material. Changes take effect for all contacts involving that material.

### Buffer format

Buffers are **definition-level** (1 float, NOT per-env) for all properties:

| Property          | dtype   | Description |
|-------------------|---------|-------------|
| `DYNAMIC_FRICTION`| float32 | kinetic friction coefficient (typically 0–1) |
| `STATIC_FRICTION` | float32 | static friction coefficient (typically 0–1) |
| `RESTITUTION`     | float32 | bounciness (0=inelastic, 1=elastic; negative=spring) |
| `DAMPING`         | float32 | implicit spring damping (only used when restitution < 0) |
| `ROUGHNESS`       | float32 | amplitude of surface roughness waves |
| `FREQUENCY`       | float32 | frequency of surface roughness waves |

### Getting a material handle

Inline vsim materials (defined with `<rigidMaterial name="...">`) have **no file path** — `get_rigid_material_paths()` returns `[]` for them. Use `create_rigid_material()` to create materials programmatically, or look them up by numeric index if you know their order.

```python
mat = v.RigidMaterial()
mat.dynamic_friction = 0.7
mat.static_friction  = 0.8
mat.restitution      = 0.0
mat_handle = env_def.create_rigid_material(mat)
```

### Usage

```python
fric_buf = torch.zeros(1, dtype=torch.float32, device=DEV)

cmd3 = env_group.create_rigid_material_property_command(
    v.RigidMaterialProperty.DYNAMIC_FRICTION,
    v.wrap_gpu_buffer(fric_buf),
    mat_handle,
)
arr3 = gym.create_gpu_array([cmd3])

# GET reads the current physics value (initialized from create_rigid_material values)
gym.get_rigid_material_properties(arr3)
print(fric_buf[0])   # e.g. 0.7

# SET: randomize friction at episode reset
fric_buf[0] = torch.rand(1).item() * 0.8 + 0.2
gym.set_rigid_material_properties(arr3)
```

### RigidMaterialProperty enum values

```python
v.RigidMaterialProperty.DYNAMIC_FRICTION  # = 0
v.RigidMaterialProperty.STATIC_FRICTION   # = 1
v.RigidMaterialProperty.RESTITUTION        # = 2
v.RigidMaterialProperty.DAMPING            # = 3
v.RigidMaterialProperty.ROUGHNESS          # = 4
v.RigidMaterialProperty.FREQUENCY          # = 5
```

---

## 4. Rigid Body Def Command (Definition Swap)

Swaps which `RigidBodyDef` is active for a rigid body instance, **per environment**. Allows morphology randomization by pre-loading multiple shape definitions and switching between them.

### Requirements

- Rigid body must be created with the **list overload** of `create_rigid_body`:
  ```python
  rb_handle = env_def.create_rigid_body([def_handle_a, def_handle_b], transform, name)
  ```
- `gym.step()` must be called at least once before `set_rigid_body_defs()`

### Buffer format

```
def_idx_buf: (N_envs,) uint32 — per-env index into the ordered def list
```

Index 0 = first def passed to `create_rigid_body([...])`, index 1 = second, etc.

### Usage

```python
# Setup (before gym.finalize)
rb_def_a = env_def.get_rigid_body_def_handle_by_name("large_box")   # def 0
rb_def_b = env_def.create_box_def(v.Vec3(0.5, 0.5, 0.5), ...)       # def 1
rb_handle = env_def.create_rigid_body([rb_def_a, rb_def_b], transform, "swappable")

# After gym.finalize + gym.step():
def_idx_buf = torch.zeros(N, dtype=torch.uint32, device=DEV)
cmd4 = env_group.create_rigid_body_def_command(
    v.wrap_gpu_buffer(def_idx_buf),
    rb_handle,
)
arr4 = gym.create_gpu_array([cmd4])

# All envs use def 0
def_idx_buf[:] = 0
gym.set_rigid_body_defs(arr4)

# Per-env morphology: env 0 gets large box, env 1 gets small box
def_idx_buf[0] = 0  # large_box
def_idx_buf[1] = 1  # small_box
gym.set_rigid_body_defs(arr4)
```

**Note:** The viewport renderer does not update for def swaps. Use the tiled RGB renderer (toggle in GUI) to see the change visually.

---

## 5. Articulation Morph Geometry — BROKEN / Unimplemented

`get_articulation_def_geometry_handle` always returns the invalid sentinel:

```
index = 0x03FFFFFF  (67108863)
type  = 63
```

This happens for every link index, before and after `gym.finalize()`, on every vsim tested (ant.vsim 9-link, minimal 2-link, vlearn 0.3.5 and 0.3.8). There is **no entry in the HTML API docs** for this function.

The intended API (unreachable):

```python
# env_def.get_articulation_def_geometry_handle(arti_def_handle, link_index)
# → always returns invalid sentinel — do not use

# env_group.create_articulation_morph_geometry_command(
#     buffer: Float32GpuBufferWrapper,        # (num_verts * 3,) — cage positions
#     vert_weight_buffer: Float32GpuBufferWrapper,   # (num_verts,) — blend weights
#     geometry_handle: GeometryHandle,        # must be valid — currently never is
#     articulation_def_handle: ArticulationDefHandle,
#     articulation_link_index: int,
#     masks_buffer: BoolGpuBufferWrapper = None,
# ) → ArticulationMorphGeometryCommand
```

`gym.get_articulation_morph_geometry_deformation_cages(arr)` and
`gym.set_articulation_morph_geometry_deformation_cages(arr, apply_smoothing=False)`
are also defined but cannot be used without a valid geometry handle.

---

## GeometryHandle

Returned by `get_rigid_body_def_geometry_handle`. Two properties:

```python
h.index()  # int — internal registry index; 0x03FFFFFF = invalid sentinel
h.type()   # int — geometry type code; 63 = invalid, 10 = morph mesh
```

---

## Summary Table

| API | Status | Buffer size | Per-env? |
|-----|--------|-------------|----------|
| `create_rigid_body_morph_geometry_command` | **Working** | `(num_verts*3,)` + `(num_verts,)` | No (def-level) |
| `create_rigid_body_property_command` (INV_MASS) | **Working (round-trip)** | `(1,)` | No |
| `create_rigid_body_property_command` (DIAG_INV_INERTIA) | **Working (round-trip)** | `(3,)` | No |
| `create_rigid_body_property_command` (BODY_TO_MODEL) | **Working (round-trip)** | `(7,)` | No |
| `create_rigid_material_property_command` | **Working** | `(1,)` | No |
| `create_rigid_body_def_command` | **Working** (needs step first) | `(N,)` uint32 | Yes |
| `create_articulation_morph_geometry_command` | **Broken** (handle always invalid) | — | — |
