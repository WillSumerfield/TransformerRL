"""Default module library: a static 8-slot torso ring + 3 effector types + 4 cap types. Owns
everything ring/limb/cap-specific (torso shape, ring-slot placement, Morphology, body assembly);
build_vsim.py stays a domain-free XML/geometry toolkit this module calls into.

Every module's joint axis is defined RELATIVE to its parent frame (one axis per type, not a table
per ring slot) -- construct_robot resolves it to an absolute world-frame value by rotating it with
the ring slot's yaw. Ring-slot geometry (_slot_dir/_slot_yaw) is computed from the slot index rather
than tabulated, since nothing here needs byte-identical reproduction of the old per-slot tables.

All physical/vocabulary constants live as private class attributes on SimpleModuleLibrary, not
module globals -- nothing outside this file (and no NEW public method beyond what the parent
ModuleLibrary interface already defines: modules/module_count/names/construct_robot) reaches them.
Callers elsewhere that need a "default" body (swing hip + knee chain, bare cap) author it with
literal type-name strings against SimpleModuleLibrary's known vocabulary, the same way
AntCodesignEnv._BASE_MORPHOLOGY already does -- that's a caller's own design choice, not data this
library needs to hand out.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple
from functools import cached_property

from task_envs.build_vsim import (
    fmt_float, vec_str, quat_axis, quat_mul, quat_str, quat_rotate, geom_volume,
    collision_xml, link_xml, joint_xml, motor_xml, sensor_xml
)
from codesigner.components.interfaces import ModuleLibrary, ModuleType, ModuleDefinition, Module, Morphology


# ---- per-module-type physical specifics ----------------------------------------------------
@dataclass
class EffectorGeometry:
    axis: Tuple[float, float, float]   # rotation axis, relative to the module's parent frame
    proximal_cyl: bool                  # True: full-length collision cyl (swing); False: shortened
                                         # distal cyl (knee/twist)


@dataclass
class CapGeom:
    kind: str
    half_dims: Tuple[float, ...]
    yaw: float                          # about the cap's own vertical, cap-frame-relative
    offset: Tuple[float, float, float]  # cap-frame-relative offset from the cap link origin


# ---- Morphology: SimpleModuleLibrary-private body representation ----------------------------
class SimpleMorphology(Morphology):
    """One robot body: per active ring slot (1-8), an ordered chain of effector TYPE NAMES plus one
    terminal cap TYPE NAME.

    module_lengths: {slot -> [len_1, len_2, ...]}, list length = # effectors. Length is currently a
                    fixed property of the effector TYPE (uniform) -- continuous length design is not
                    implemented.
    effector_types: {slot -> [name, ...]} parallel to module_lengths; defaults to the library's
                    canonical chain (swing hip, knee joints), reproducing the original ant body.
    cap_types:      {slot -> name}; defaults to the library's canonical ("bare") cap.
    A limb is present iff it has >=1 effector.
    """

    def __init__(self, modules: list[list[Module]]):
        super().__init__(modules)

    @cached_property
    def limbs(self) -> frozenset:
        return frozenset(n for n, L in self.module_lengths.items() if L)

    def num_modules(self, n: int) -> int:
        return sum(self._modules)

    def cap_of(self, n: int) -> str:
        return self.cap_types.get(n)


class SimpleModuleLibrary(ModuleLibrary):

    # ---- Geometry -------------------------------------------------------------
    _EFF_LENGTH = 0.65
    _TORSO_RADIUS = 0.3
    _TORSO_STUB = 0.15
    _LEG_CYL_FRAC = _TORSO_STUB * 2 / _EFF_LENGTH

    _MAX_LIMB_LENGTH = 4
    _MAX_EFFECTORS = _MAX_LIMB_LENGTH - 1

    _CANON_EFF_HIP = "swing"
    _CANON_EFF_JOINT = "knee"
    _CANON_CAP = "bare"

    _EFFORT = "3.4e+38"
    _DENSITY = "5.0"
    _RADIUS = "0.08"
    _JOINT_DYNAMICS = 'damping="1.0" stiffness="100" friction="0.0" armature="1.0"'
    _JOINT_VELOCITY = "30.0"

    _PAD_HALF_LEN = 0.10
    _LONG_EFF_VOLUME = math.pi * float(_RADIUS) ** 2 * 2 * (_LEG_CYL_FRAC * _EFF_LENGTH)
    # Every cap masses the same: half a long (knee/twist) effector module. See build_ant_vsim's old
    # cap-geometry docstring for why (caps span a >10x mass range by shape alone otherwise).
    _CAP_MASS = 0.5 * float(_DENSITY) * _LONG_EFF_VOLUME
    _KNEE_NOMINAL_BEND = 1.134464045365651486

    _ROOT_AXIS_DIR = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}

    _MODULE_NAMES = ('swing', 'knee', 'twist', 'bare', 'foot', 'pad', 'ball')
    _DEFAULT_ORIENTATIONS = [0]
    _CONNECTIONS_ALL = {0: {}}
    for _name in _MODULE_NAMES:
        _CONNECTIONS_ALL[0][_name] = _DEFAULT_ORIENTATIONS
    del _name


    def __init__(self, root_axes: list = None, root_axis_range: tuple = (-1.0, 1.0)):
        """root_axes: None -> free-floating base. [] -> world-mounted, fixed. non-empty list of
        'x'/'y'/'z' -> world-mounted with one actuated prismatic joint per named axis, ranged over
        root_axis_range (meters, shared across every configured axis)."""
        self.root_axes = root_axes
        self.root_axis_range = root_axis_range
        super().__init__(name="simple")

    # ------------------------- Module Definitions -------------------------------

    def _modules(self) -> List[ModuleDefinition]:
        return [
            self._define_effector("swing", [(-0.7, 0.7)], EffectorGeometry(axis=self._ROOT_AXIS_DIR["z"], proximal_cyl=True)),
            self._define_effector("knee", [(0.5, 1.5)], EffectorGeometry(axis=self._ROOT_AXIS_DIR["y"], proximal_cyl=False)),
            self._define_effector("twist", [(-1.5, 1.5)], EffectorGeometry(axis=self._ROOT_AXIS_DIR["x"], proximal_cyl=False)),
            self._define_cap("bare", geometry=None),
            self._define_cap("foot", geometry=[
                CapGeom("cylinder", (0.25, 0.03), 0.25 * math.pi, (0.0, 0.0, 0.0)),
                CapGeom("cylinder", (0.25, 0.03), -0.25 * math.pi, (0.0, 0.0, 0.0)),
            ]),
            self._define_cap("pad", geometry=[
                CapGeom("box", (self._PAD_HALF_LEN, 0.10, 0.02), 0.0, (0.6 * self._PAD_HALF_LEN, 0.0, 0.0)),
            ]),
            self._define_cap("ball", geometry=[
                CapGeom("sphere", (0.09,), 0.0, (0.0, 0.0, 0.0)),
            ]),
        ]

    def _define_cap(self, name: str, geometry) -> ModuleDefinition:
        return ModuleDefinition(
            name=name,
            type=ModuleType.CAP,
            valid_orientations=self._DEFAULT_ORIENTATIONS,
            action_dimensions=0,
            joint_limits=None,
            geometry=geometry,
            valid_configurations=None,
        )

    def _define_effector(self, name: str, joint_limits: List[Tuple[float, float]], geometry) -> ModuleDefinition:
        return ModuleDefinition(
            name=name,
            type=ModuleType.EFFECTOR,
            valid_orientations=self._DEFAULT_ORIENTATIONS,
            action_dimensions=1,
            joint_limits=joint_limits,
            geometry=geometry,
            valid_configurations=self._CONNECTIONS_ALL,
        )

    # ------------------------- Robot construction -------------------------------

    def construct_robot(self, configuration: Morphology) -> str:
        root_axes = self.root_axes or []
        parts = ['<robot name="torso">']
        for i in range(len(root_axes)):
            parts.append(self._root_anchor_link_xml(f"root_mount_{i}"))

        limbs = []  # (slot, effectors: List[Module], cap: Module | None)
        for chain in configuration:
            n = chain[0].orientation
            effectors = [m for m in chain if m.module_type.type == ModuleType.EFFECTOR]
            cap = next((m for m in chain if m.module_type.type == ModuleType.CAP), None)
            limbs.append((n, effectors, cap))
        active = sorted(n for n, _, _ in limbs)
        by_n = {n: (effs, cap) for n, effs, cap in limbs}
        maxd = max((len(effs) for _, effs, _ in limbs), default=0)

        torso_collisions = [collision_xml("0.0 0.0 0.0", "0.0 0.0 0.0 1.0", "sphere", (0.25,))]
        for n in active:
            torso_collisions.append(collision_xml(vec_str(self._TORSO_STUB, self._slot_dir(n)), self._slot_rot_str(n),
                                                    "cylinder", (self._TORSO_STUB, float(self._RADIUS))))
        parts.append(link_xml("torso", self._DENSITY, torso_collisions))

        # Links declared DEPTH-MAJOR, ascending active.
        for d in range(1, maxd + 1):
            for n in active:
                effs, _ = by_n[n]
                if len(effs) >= d:
                    parts.append(self._effector_link_xml(n, d, effs[d - 1]))

        # Cap links AFTER every effector link: caps attach via FIXED joints (0 DOF), so their
        # declaration order cannot perturb vsim's DOF enumeration.
        for n in active:
            effs, cap = by_n[n]
            if cap is not None:
                n_knees = sum(1 for m in effs if m.module_type.name == "knee")
                parts.append(self._cap_link_xml(n, len(effs) + 1, cap, n_knees))

        # Joints declared DEPTH-MAJOR, reverse-active within each depth.
        for d in range(1, maxd + 1):
            for n in reversed(active):
                effs, _ = by_n[n]
                if len(effs) >= d:
                    parts.append(self._effector_joint_xml(n, d, effs[d - 1]))

        for n in active:
            effs, cap = by_n[n]
            if cap is not None:
                parts.append(self._cap_joint_xml(n, len(effs) + 1))

        if root_axes:
            lo, hi = self.root_axis_range
            chain_parents = [f"root_mount_{i}" for i in range(len(root_axes))]
            chain_children = [f"root_mount_{i + 1}" for i in range(len(root_axes) - 1)] + ["torso"]
            for axis, parent, child in zip(root_axes, chain_parents, chain_children):
                parts.append(self._root_joint_xml(axis, parent, child, lo, hi))

        # Motors declared per-limb (n ascending, then depth), then root-axis motors.
        parts.append('    <actuator>')
        for n in active:
            effs, _ = by_n[n]
            for i in range(1, len(effs) + 1):
                parts.append(motor_xml(f"joint_{n}_{i}", f"joint_{n}_{i}", "150.0", "-1.0", "1.0"))
        for axis in root_axes:
            parts.append(motor_xml(f"root_{axis}", f"root_{axis}", "150.0", "-1.0", "1.0"))
        parts.append('    </actuator>')

        # ONE contact sensor per limb, ascending-active. On the CAP when it's a real body, else on
        # the terminal effector.
        if active:
            parts.append('    <forceSensor>')
            for n in active:
                effs, cap = by_n[n]
                k = len(effs)
                if cap is None:
                    offset = vec_str(self._EFF_LENGTH, self._slot_dir(n))
                    parts.append(sensor_xml(f"mod_{n}_{k}_sensor", f"mod_{n}_{k}", offset))
                else:
                    parts.append(sensor_xml(f"cap_{n}_{k + 1}_sensor", f"cap_{n}_{k + 1}", "0.0 0.0 0.0"))
            parts.append('    </forceSensor>')

        parts.append('</robot>')
        return '\n'.join(parts)

    def _cap_frame(self, n: int, n_knees: int) -> tuple:
        """Cap frame: yaw to the ring slot, then pitch UP by the nominal knee bend so a bare-standing
        cap's contact face lands horizontal (caps ride a FIXED joint, so this compensation is baked
        statically into the cap's own geometry rather than tracked at runtime)."""
        pitch = min(self._cap_pitch() * n_knees, 0.5 * math.pi)
        return quat_mul(self._slot_yaw(n), quat_axis((0.0, 1.0, 0.0), -pitch))

    def _effector_link_xml(self, n: int, i: int, module: Module) -> str:
        geom = module.module_type.geometry
        half = self._EFF_LENGTH / 2 if geom.proximal_cyl else self._LEG_CYL_FRAC * self._EFF_LENGTH
        coll = collision_xml(vec_str(half, self._slot_dir(n)), self._slot_rot_str(n), "cylinder",
                              (half, float(self._RADIUS)))
        return link_xml(f"mod_{n}_{i}", self._DENSITY, [coll])

    def _effector_joint_xml(self, n: int, i: int, module: Module) -> str:
        geom = module.module_type.geometry
        lo, hi = module.module_type.joint_limits[0]
        if i == 1:
            parent, origin = "torso", vec_str(self._TORSO_RADIUS, self._slot_dir(n))
        else:
            parent, origin = f"mod_{n}_{i - 1}", vec_str(self._EFF_LENGTH, self._slot_dir(n))
        axis = vec_str(1.0, quat_rotate(self._slot_yaw(n), geom.axis))
        return joint_xml(f"joint_{n}_{i}", "revolute", parent, f"mod_{n}_{i}", origin,
                          axis=axis, limits=(fmt_float(lo), fmt_float(hi)),
                          effort=self._EFFORT, velocity=self._JOINT_VELOCITY, dynamics=self._JOINT_DYNAMICS)

    def _cap_link_xml(self, n: int, d: int, cap: Module, n_knees: int) -> str:
        geoms = cap.module_type.geometry
        density = self._CAP_MASS / sum(geom_volume(g.kind, g.half_dims) for g in geoms)
        frame = self._cap_frame(n, n_knees)
        collisions = []
        for g in geoms:
            rot = quat_str(quat_mul(frame, quat_axis((0.0, 0.0, 1.0), g.yaw)))
            origin = vec_str(1.0, quat_rotate(frame, g.offset))
            collisions.append(collision_xml(origin, rot, g.kind, g.half_dims))
        return link_xml(f"cap_{n}_{d}", density, collisions)

    def _cap_joint_xml(self, n: int, d: int) -> str:
        origin = vec_str(self._EFF_LENGTH, self._slot_dir(n))
        return joint_xml(f"capjoint_{n}_{d}", "fixed", f"mod_{n}_{d - 1}", f"cap_{n}_{d}", origin)

    def _root_anchor_link_xml(self, name: str) -> str:
        coll = collision_xml("0.0 0.0 0.0", "0.0 0.0 0.0 1.0", "sphere", (0.05,))
        return link_xml(name, self._DENSITY, [coll])

    def _root_joint_xml(self, axis: str, parent: str, child: str, lo: float, hi: float) -> str:
        return joint_xml(f"root_{axis}", "prismatic", parent, child, "0.0 0.0 0.0",
                          axis=vec_str(1.0, self._ROOT_AXIS_DIR[axis]), limits=(fmt_float(lo), fmt_float(hi)),
                          effort=self._EFFORT, velocity=self._JOINT_VELOCITY, dynamics=self._JOINT_DYNAMICS)

    def _slot_dir(self, n: int) -> tuple:
        """Unit outward direction for ring slot n (1-8)."""
        theta = (n - 1) * 0.25 * math.pi
        return (math.cos(theta), math.sin(theta), 0.0)

    def _slot_yaw(self, n: int) -> tuple:
        """Quaternion yawing the canonical (slot-1) frame to slot n."""
        return quat_axis((0.0, 0.0, 1.0), (n - 1) * 0.25 * math.pi)

    def _slot_rot_str(self, n: int) -> str:
        return quat_str(self._slot_yaw(n))
