"""vsim body builder — variable-length limbs (module chains) with typed modules.

Phase 1 (grill 2026-07-02): a limb = a chain of actuated modules along the outward unit dir
u=_DIR[n]; joint origin of module i = (torso_radius if i==1 else length[i-1]) * u; names
aux_n/leg_n -> mod_{n}_{i}, hip_n/ankle_n -> joint_{n}_{i}, "torso" root stays.

Phase 5a stages 1+2 (grill 2026-07-20) makes module TYPE a designed field:
  - EFFECTOR types (swing / knee / twist) = (local joint axis, limits, fixed default length).
    Position-independent in the limb-local frame; see _EFF_LENGTH for the derivation.
  - CAP types (bare / foot / pad / ball) = a passive terminal module on a FIXED joint. Bare is
    zero-morphology (no link; contact stays on the terminal effector) so the canonical
    swing-then-knee chain with a bare cap still reproduces the phase-1 ant EXACTLY.
  - The grammar forces a cap at the deepest slot => at most MAX_EFFECTORS = MAX_LIMB_LENGTH-1
    effectors per limb. Caps never actuate, so effector <-> DOF stays 1:1.
  - The contact force-sensor moves to the cap when the cap is a real body (one sensor per limb
    either way, emitted ascending-active as ant_multimorph's sensor_indices assumes).

OPEN / defer to branch-time runtime check: the joint DECLARATION ORDER below assumes vsim's
reverse-DFS still yields a depth-major DOF index layout ((d-1)*8 + (n-1)). That coupling to the
scatter in ant_multimorph can only be confirmed by loading a built body in the sim. Flagged.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

from transformer_rl.vocab import (EFF_SWING, EFF_KNEE, EFF_TWIST, CAP_BARE, CAP_FOOT, CAP_PAD,
                                  CAP_BALL, canonical_eff, CANON_CAP)

_GENERATED_DIR = Path(__file__).parent / "assets" / "generated"

MAX_LIMB_LENGTH = 4
# Phase 5: the constrained decoder forces the DEEPEST slot to a cap, so a limb holds at most
# MAX_LIMB_LENGTH-1 effectors plus one cap. The slot layout still spans MAX_LIMB_LENGTH (the cap
# rides the depth==count slot), so obs/DOF padding is unchanged.
MAX_EFFECTORS = MAX_LIMB_LENGTH - 1

_EFFORT  = "3.40282347e+38"
_DENSITY = "5.0"
_RADIUS  = "0.08"

DEFAULT_HIP   = 0.282842712474619062
DEFAULT_ANKLE = 0.632455532033675882
HIP_RANGE     = (0.5 * DEFAULT_HIP,   1.5 * DEFAULT_HIP)
ANKLE_RANGE   = (0.5 * DEFAULT_ANKLE, 1.5 * DEFAULT_ANKLE)

_TORSO_RADIUS = 0.282842712474619062
_TORSO_STUB   = 0.141421356237309531
_LEG_CYL_FRAC = _TORSO_STUB * 2 / DEFAULT_ANKLE

_S = math.sqrt(0.5)
_DIR = {
    1: (1.0, 0.0, 0.0),  2: (_S,  _S, 0.0),  3: (0.0, 1.0, 0.0),  4: (-_S,  _S, 0.0),
    5: (-1.0, 0.0, 0.0), 6: (-_S, -_S, 0.0), 7: (0.0, -1.0, 0.0), 8: (_S,  -_S, 0.0),
}
_CYL_ROT = {
    1: "0.0 0.0 0.0 1.0",
    2: "0.0 0.0 0.382683432365089782 0.923880",
    3: "0.0 0.0 0.707106781186547573 0.707106781186547573",
    4: "0.0 0.0 0.923879532511286738 0.382683",
    5: "0.0 0.0 1.0 0.0",
    6: "0.0 0.0 -0.923879532511286738 0.382683",
    7: "0.0 0.0 -0.707106781186547573 0.707106781186547573",
    8: "0.0 0.0 -0.382683432365089782 0.923880",
}
# Numeric form of _CYL_ROT: limb n is yawed (n-1)*45deg about model z, wrapped to (-180, 180].
# _CYL_ROT stays a literal table so canonical bodies remain byte-identical to phase 1; this is used
# only by the caps, which no phase before 5 emitted.
_YAW = {n: (((n - 1) * 0.25 * math.pi + math.pi) % (2 * math.pi)) - math.pi for n in range(1, 9)}
_D632 = 0.632455532033675882
_D447 = 0.447486996650695801
_SENSOR_BASE = {
    1: (_D632, 0.0, 0.0),   2: (_D447, _D447, 0.0),  3: (0.0, _D632, 0.0),  4: (-_D447, _D447, 0.0),
    5: (-_D632, 0.0, 0.0),  6: (-_D447, -_D447, 0.0), 7: (0.0, -_D632, 0.0), 8: (_D447, -_D447, 0.0),
}
_ANKLE_AXIS = {
    1: "0.0 1.0 0.0",  2: "-1.0 1.0 0.0", 3: "-1.0 0.0 0.0", 4: "1.0 1.0 0.0",
    5: "0.0 -1.0 0.0", 6: "-1.0 1.0 0.0", 7: "1.0 0.0 0.0",  8: "1.0 1.0 0.0",
}
_POS_LIMITS = ("0.523598790168762207", "1.745329300562540764")
_NEG_LIMITS = ("-1.745329300562540764", "-0.523598790168762207")
_ANKLE_LIMITS = {n: (_NEG_LIMITS if n in (4, 6) else _POS_LIMITS) for n in range(1, 9)}
_HIP_LIMITS = ("-0.698131720225016239", "0.698131720225016239")
# Twist (5a): rotation ABOUT the limb axis. Provisional +-90deg — tunable, revisit on the tune sweep.
TWIST_LIMIT = 1.5707963267948966
_TWIST_LIMITS = (repr(-TWIST_LIMIT), repr(TWIST_LIMIT))
# --- Phase-5 effector types: (local joint axis, limits, fixed default length) ------------------
# Position-INDEPENDENT in the limb-local frame (x = along-limb outward, y = tangent, z = up); the
# model-frame axis is that local axis rotated by _CYL_ROT[n]. Verified against the phase-1 tables:
#   local z -> (0,0,1)                 (invariant under R_z)     == the old hip/swing axis
#   local y -> R_z(theta_n).(0,1,0)    == _ANKLE_AXIS[n]         == the old ankle/knee axis
#   local x -> R_z(theta_n).(1,0,0)    == _DIR[n]                == along-limb, the new twist axis
# The knee's sign flip on limbs 4 and 6 is decoupled per-limb HANDEDNESS (negated axis absorbed into
# negated limits), NOT a property of the type — hence _ANKLE_LIMITS stays a per-limb table.
_EFF_LENGTH = {EFF_SWING: DEFAULT_HIP, EFF_KNEE: DEFAULT_ANKLE, EFF_TWIST: DEFAULT_ANKLE}
# proximal cylinder convention (full-length cyl, == the old aux/hip link) vs distal (_LEG_CYL_FRAC
# shortened, == the old leg/ankle link). Keyed by TYPE now, not depth, so the canonical
# swing-then-knee chain still emits byte-identical geometry.
_EFF_PROXIMAL_CYL = {EFF_SWING: True, EFF_KNEE: False, EFF_TWIST: False}

# --- Phase-5 cap types: passive terminal bodies (no joint DOF, no motor) -----------------------
# Dims are HALF-EXTENTS (vsim convention: create_box_def takes half_size, and <cylinder length> is a
# half-length — see _cyl, which spans a module by passing length/2).
#
# Caps are GROUND-FACING: their frame is the limb's yaw composed with a pitch that undoes the knee
# bend, so the contact face is horizontal in nominal stance (see _cap_rot). Cap-frame axes are then
# x = outward along the limb (horizontal), y = tangent, z = world-up.
#   foot — two cylinders crossed in an X, lying flat on the ground
#   pad  — square plate lying flat on the ground
#   ball — sphere; point contact, orientation irrelevant
# Each entry lists (kind, half-dims, yaw-about-cap-z, cap-frame offset) so one cap can carry several
# geoms. The pad is offset OUTWARD so the limb meets it one fifth along from its heel rather than at
# its centre: the plate reaches forward of the contact point (radially away from the torso), like a
# foot rather than a coaster. CAP_BARE has NO entry: it is the zero-morphology cap (no link, no
# joint, contact stays on the terminal effector) and reproduces the phase-1 body exactly.
_PAD_HALF_LEN = 0.10
_CAP_GEOM = {
    CAP_FOOT: [("cylinder", (0.25, 0.03),  0.25 * math.pi, (0.0, 0.0, 0.0)),
               ("cylinder", (0.25, 0.03), -0.25 * math.pi, (0.0, 0.0, 0.0))],
    CAP_PAD:  [("box", (_PAD_HALF_LEN, 0.10, 0.02), 0.0, (0.6 * _PAD_HALF_LEN, 0.0, 0.0))],
    CAP_BALL: [("sphere", (0.09,), 0.0, (0.0, 0.0, 0.0))],
}
# Nominal knee bend the cap pitch cancels: midpoint of the knee range (_POS_LIMITS). A cap only sits
# truly flat at this one pose — it is welded to the limb, so it pitches with the gait. Each KNEE in
# the chain bends the frame further, hence the per-knee multiple (clamped upright at 90deg). Swing
# and twist do not pitch the limb, so they do not contribute.
_CAP_PITCH = 0.5 * (float(_POS_LIMITS[0]) + float(_POS_LIMITS[1]))
# Every cap masses the same: half a long effector (knee/twist) module. Equal mass is deliberate —
# caps span a >10x mass range if left at the body density, and a cap rides the tip of a swinging
# limb where it dominates rotational inertia, so the generator would select on mass rather than on
# contact geometry. Realized as a per-cap density (mass/volume) since _INERTIAL_BLOCK derives mass
# from density. Crossed cylinders double-count their intersection (~10% of foot volume) if vsim
# sums geom volumes rather than unioning them.
_LONG_EFF_VOLUME = math.pi * float(_RADIUS) ** 2 * 2 * (_LEG_CYL_FRAC * DEFAULT_ANKLE)
_CAP_MASS = 0.5 * float(_DENSITY) * _LONG_EFF_VOLUME


def _geom_volume(kind: str, dims: tuple) -> float:
    if kind == "box":
        return 8.0 * dims[0] * dims[1] * dims[2]
    if kind == "cylinder":
        return math.pi * dims[1] ** 2 * 2 * dims[0]
    if kind == "sphere":
        return 4.0 / 3.0 * math.pi * dims[0] ** 3
    raise ValueError(f"unknown geom kind {kind}")


def _cap_density(ctype: int) -> float:
    """Density giving every cap type the same _CAP_MASS regardless of its shape."""
    return _CAP_MASS / sum(_geom_volume(k, d) for k, d, _, _ in _CAP_GEOM[ctype])


_JOINT_DYNAMICS = 'damping="1.0" stiffness="100" friction="0.0" armature="1.0"'
_JOINT_VELOCITY = "30.0"


def _inertial(density) -> str:
    """mass=-1 => vsim derives mass and inertia from the collision volume at this density."""
    return f"""
        <inertial>
            <origin xyz="0.0 0.0 0.0" rot="0.0 0.0 0.0 1.0"/>
            <mass value="-1.0"/>
            <density value="{density}"/>
            <inertia ixx="-1.0" iyy="-1.0" izz="-1.0" ixy="-1.0" ixz="-1.0" iyz="-1.0"/>
        </inertial>"""


_INERTIAL_BLOCK = _inertial(_DENSITY)   # torso + every effector link


@dataclass
class Morphology:
    """One robot body: per active limb, an ordered chain of EFFECTORS plus one terminal CAP.

    module_lengths: {limb 1..8 -> [len_1, len_2, ...]}, list length = # effectors (1..MAX_EFFECTORS).
                    Length is a fixed property of the effector TYPE (not designed yet).
    effector_types: {limb -> [type id, ...]} parallel to module_lengths; defaults to the canonical
                    swing-then-knee chain, which reproduces the phase-1 body exactly.
    cap_types:      {limb -> cap type id}; defaults to CAP_BARE (zero-morphology == phase 1).
    A limb is present iff it has >=1 effector.
    """
    module_lengths: dict                        # limb -> list[float]
    effector_types: dict = field(default_factory=dict)   # limb -> list[int]
    cap_types: dict = field(default_factory=dict)        # limb -> int

    def __post_init__(self):
        for n, L in self.module_lengths.items():
            if not L:
                continue
            self.effector_types.setdefault(n, [canonical_eff(d) for d in range(len(L))])
            self.cap_types.setdefault(n, CANON_CAP)

    @property
    def legs(self) -> frozenset:
        return frozenset(n for n, L in self.module_lengths.items() if L)

    def num_modules(self, n: int) -> int:
        return len(self.module_lengths.get(n, ()))

    def cap_of(self, n: int) -> int:
        return self.cap_types.get(n, CANON_CAP)

    @classmethod
    def from_legs(cls, legs) -> "Morphology":
        """Default-length, length-2 body for a bare limb set (reproduces the current fixed ant)."""
        return cls({n: [DEFAULT_HIP, DEFAULT_ANKLE] for n in frozenset(legs)})

    @classmethod
    def from_counts(cls, counts: dict) -> "Morphology":
        """Canonical-type body from per-limb effector counts {limb -> k}: swing then knees, bare cap.
        Phase-1-equivalent (that phase designed COUNT only)."""
        return cls.from_design({n: [canonical_eff(d) for d in range(k)]
                                for n, k in counts.items() if k > 0}, {})

    @classmethod
    def from_design(cls, effector_types: dict, cap_types: dict) -> "Morphology":
        """Body from the generator's designed types: {limb -> [effector type ids]} + {limb -> cap
        type id}. Per-module length is derived from the effector TYPE (continuous length design
        stays deferred to Phase 10)."""
        et = {n: list(ts) for n, ts in effector_types.items() if ts}
        return cls({n: [_EFF_LENGTH[t] for t in ts] for n, ts in et.items()},
                   et, {n: cap_types.get(n, CANON_CAP) for n in et})

    # Back-compat shim so callers reading the old fields keep working during migration.
    @property
    def hip_lengths(self) -> dict:
        return {n: L[0] for n, L in self.module_lengths.items() if L}

    @property
    def ankle_lengths(self) -> dict:
        return {n: L[1] for n, L in self.module_lengths.items() if len(L) > 1}


def _f(v: float) -> str:
    return repr(0.0 if v == 0 else float(v))


def _vec(mag: float, u: tuple) -> str:
    return f"{_f(mag * u[0])} {_f(mag * u[1])} {_f(mag * u[2])}"


def _qaxis(axis: tuple, angle: float) -> tuple:
    s = math.sin(0.5 * angle)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle))


def _qmul(a: tuple, b: tuple) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qstr(q: tuple) -> str:
    return f"{_f(q[0])} {_f(q[1])} {_f(q[2])} {_f(q[3])}"


def _qrot(q: tuple, u: tuple) -> tuple:
    x, y, z, w = q
    t = (2 * (y * u[2] - z * u[1]), 2 * (z * u[0] - x * u[2]), 2 * (x * u[1] - y * u[0]))
    c = (y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0])
    return tuple(u[i] + w * t[i] + c[i] for i in range(3))


def _cap_frame(n: int, n_knees: int) -> tuple:
    """Cap frame in the link frame: yaw to the limb, then pitch UP by the nominal knee bend so the
    contact face lands horizontal. Its axes are x = outward along the limb, y = tangent, z = up.

    The pitch cancels the knee rotation because the cap link rides the terminal effector's frame,
    which the knee has already pitched down by ~_CAP_PITCH per knee. Limbs 4 and 6 need no special
    case: their ankle axis AND their limits are both negated (_NEG_LIMITS), so the frame ends up
    rotated the same way as every other limb.
    """
    pitch = min(_CAP_PITCH * n_knees, 0.5 * math.pi)
    return _qmul(_qaxis((0.0, 0.0, 1.0), _YAW[n]), _qaxis((0.0, 1.0, 0.0), -pitch))


def _cap_rot(n: int, n_knees: int, geom_yaw: float) -> str:
    """Orientation of one cap geom: the cap frame, then the geom's own yaw about the cap vertical."""
    return _qstr(_qmul(_cap_frame(n, n_knees), _qaxis((0.0, 0.0, 1.0), geom_yaw)))


def _cap_off(n: int, n_knees: int, off: tuple) -> str:
    """A cap-frame offset expressed in the link frame (the geom yaw does not move the centre)."""
    return " ".join(_f(c) for c in _qrot(_cap_frame(n, n_knees), off))


def _cyl(center: float, half_len: float, u: tuple, rot: str) -> str:
    return f"""        <collision>
            <origin xyz="{_vec(center, u)}" rot="{rot}"/>
            <geometry>
                <cylinder length="{_f(half_len)}" radius="{_RADIUS}"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>"""


def _effector_axis_limits(n: int, etype: int) -> tuple:
    """(axis, (lo, hi)) in MODEL frame for effector type `etype` on limb n. See _EFF_LENGTH."""
    if etype == EFF_SWING:
        return "0.0 0.0 1.0", _HIP_LIMITS
    if etype == EFF_KNEE:
        return _ANKLE_AXIS[n], _ANKLE_LIMITS[n]
    if etype == EFF_TWIST:
        return _vec(1.0, _DIR[n]), _TWIST_LIMITS
    raise ValueError(f"unknown effector type {etype}")


def _module_link(n: int, i: int, length: float, etype: int) -> str:
    """Link for effector i (1-based) of limb n. The cylinder convention follows the TYPE: swing uses
    the proximal (aux/hip) full-length cyl, knee/twist the distal (leg/ankle) _LEG_CYL_FRAC cyl."""
    u, rot = _DIR[n], _CYL_ROT[n]
    if _EFF_PROXIMAL_CYL[etype]:
        cyl = _cyl(length / 2, length / 2, u, rot)          # == old aux_link(H)
    else:
        cyl = _cyl(_LEG_CYL_FRAC * length, _LEG_CYL_FRAC * length, u, rot)  # == old leg_link(A)
    return f"""    <link name="mod_{n}_{i}">{_INERTIAL_BLOCK}
{cyl}
    </link>"""


def _module_joint(n: int, i: int, prev_length: float, etype: int) -> str:
    """Joint feeding effector i of limb n. i==1 hangs off the torso at torso radius; i>=2 hangs off
    mod_{n}_{i-1} at that module's length. Axis + limits come from the effector TYPE."""
    if i == 1:
        parent, origin = "torso", _vec(_TORSO_RADIUS, _DIR[n])
    else:
        parent, origin = f"mod_{n}_{i-1}", _vec(prev_length, _DIR[n])
    axis, (lo, hi) = _effector_axis_limits(n, etype)
    return f"""    <joint name="joint_{n}_{i}" type="revolute">
        <child link="mod_{n}_{i}"/>
        <parent link="{parent}"/>
        <origin xyz="{origin}" rot="0.0 0.0 0.0 1.0"/>
        <linkTransform xyz="{origin}" rot="0.0 0.0 0.0 1.0"/>
        <axis xyz="{axis}"/>
        <limit lower="{lo}" upper="{hi}" effort="{_EFFORT}" velocity="{_JOINT_VELOCITY}"/>
        <dynamics {_JOINT_DYNAMICS}/>
    </joint>"""


def _motor(n: int, i: int) -> str:
    return f'        <motor name="joint_{n}_{i}" joint="joint_{n}_{i}" gear="150.0" lowLimit="-1.0" highLimit="1.0"/>'


# ---- caps: passive terminal bodies (fixed joint -> 0 DOF, no motor, no action) ----------------
def _cap_geom_xml(kind: str, dims: tuple) -> str:
    if kind == "box":
        return f'<box size="{_f(dims[0])} {_f(dims[1])} {_f(dims[2])}"/>'
    if kind == "cylinder":
        return f'<cylinder length="{_f(dims[0])}" radius="{_f(dims[1])}"/>'
    return f'<sphere radius="{_f(dims[0])}"/>'


def _cap_link(n: int, d: int, ctype: int, n_knees: int) -> str:
    """Cap link at 1-based slot depth d of limb n (d == #effectors + 1, the slot the cap rides).

    Geoms sit at the limb tip unless _CAP_GEOM gives them a cap-frame offset (the pad reaches
    outward). Density is per-type so every cap masses the same (_CAP_MASS) whatever its volume.
    """
    parts = [f'    <link name="cap_{n}_{d}">{_inertial(_cap_density(ctype))}']
    for kind, dims, yaw, off in _CAP_GEOM[ctype]:
        parts.append(f"""        <collision>
            <origin xyz="{_cap_off(n, n_knees, off)}" rot="{_cap_rot(n, n_knees, yaw)}"/>
            <geometry>
                {_cap_geom_xml(kind, dims)}
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>""")
    parts.append("    </link>")
    return "\n".join(parts)


def _cap_joint(n: int, d: int, term_length: float) -> str:
    """FIXED (0-DOF) joint attaching limb n's cap to its terminal effector — a cap never actuates,
    so effector <-> DOF stays 1:1 and vsim's DOF enumeration is unchanged."""
    origin = _vec(term_length, _DIR[n])
    return f"""    <joint name="capjoint_{n}_{d}" type="fixed">
        <child link="cap_{n}_{d}"/>
        <parent link="mod_{n}_{d-1}"/>
        <origin xyz="{origin}" rot="0.0 0.0 0.0 1.0"/>
        <linkTransform xyz="{origin}" rot="0.0 0.0 0.0 1.0"/>
    </joint>"""


def _sensor(n: int, k: int, term_length: float) -> str:
    """Foot/contact sensor on the terminal effector k of limb n, scaled by its length (bare cap)."""
    r = term_length / DEFAULT_ANKLE
    b = _SENSOR_BASE[n]
    off = f"{_f(r * b[0])} {_f(r * b[1])} {_f(r * b[2])}"
    return f'        <sensor name="mod_{n}_{k}_sensor" link="mod_{n}_{k}" offset="{off}" flags="contact "/>'


def _cap_sensor(n: int, d: int) -> str:
    """Contact sensor RELOCATED onto a morphology cap (the cap is the part that touches ground).
    Cap geoms are centred on the link origin, so the sensor sits there too."""
    return (f'        <sensor name="cap_{n}_{d}_sensor" link="cap_{n}_{d}" '
            f'offset="0.0 0.0 0.0" flags="contact "/>')


def build_ant_vsim(morph: Morphology) -> str:
    active = sorted(morph.legs)
    L = morph.module_lengths
    maxd = max((len(L[n]) for n in active), default=0)
    parts = ['<robot name="torso">']

    parts.append('    <link name="torso">')
    parts.append(_INERTIAL_BLOCK)
    parts.append("""        <collision>
            <origin xyz="0.0 0.0 0.0" rot="0.0 0.0 0.0 1.0"/>
            <geometry>
                <sphere radius="0.25"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>""")
    for n in active:
        parts.append(_cyl(_TORSO_STUB, _TORSO_STUB, _DIR[n], _CYL_ROT[n]))
    parts.append('    </link>')

    T = morph.effector_types
    # Links declared DEPTH-MAJOR, ascending active — matches the old "all aux, then all leg" order
    # so a length-2 body is byte-identical (modulo names) to the current ant.
    for d in range(1, maxd + 1):
        for n in active:
            if len(L[n]) >= d:
                parts.append(_module_link(n, d, L[n][d - 1], T[n][d - 1]))

    # Cap links AFTER every module link: caps attach via FIXED joints (0 DOF), so their declaration
    # order cannot perturb vsim's DOF enumeration; links are looked up by name anyway.
    for n in active:
        c = morph.cap_of(n)
        if c != CAP_BARE:
            parts.append(_cap_link(n, len(L[n]) + 1, c, T[n].count(EFF_KNEE)))

    # Joints declared DEPTH-MAJOR, reverse-active within each depth (extends the old
    # "reversed hips, then reversed ankles" so reverse-DFS yields depth-major ascending DOFs).
    # RUNTIME-VERIFY at branch time that this reproduces the length-2 DOF ordering.
    for d in range(1, maxd + 1):
        for n in reversed(active):
            if len(L[n]) >= d:
                prev_len = L[n][d - 2] if d >= 2 else 0.0
                parts.append(_module_joint(n, d, prev_len, T[n][d - 1]))

    for n in active:
        if morph.cap_of(n) != CAP_BARE:
            k = len(L[n])
            parts.append(_cap_joint(n, k + 1, L[n][k - 1]))

    # Motors declared per-limb (n ascending, then depth) — matches the old actuator order so the
    # length-2 body is byte-identical (modulo names) to the current ant. vsim maps motor->joint by
    # name, so XML order is not load-bearing, but we keep it identical to remove all doubt.
    parts.append('    <actuator>')
    for n in active:
        for i in range(1, len(L[n]) + 1):
            parts.append(_motor(n, i))
    parts.append('    </actuator>')

    # ONE contact sensor per limb, ascending-active (ant_multimorph's sensor_indices assumes that
    # order). It sits on the CAP when the cap is a real body, else on the terminal effector.
    if active:
        parts.append('    <forceSensor>')
        for n in active:
            k = len(L[n])
            c = morph.cap_of(n)
            parts.append(_sensor(n, k, L[n][k - 1]) if c == CAP_BARE else _cap_sensor(n, k + 1))
        parts.append('    </forceSensor>')

    parts.append('</robot>')
    return '\n'.join(parts)


def write_vsim(morph: Morphology, index: int) -> Path:
    """Write a morphology's vsim XML to a deterministic per-index path (no caching; always rewrites)."""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _GENERATED_DIR / f"ant_morph_{index}.vsim"
    path.write_text(build_ant_vsim(morph))
    return path


def designs_from_arrays(counts_2d, eff_sub_3d, cap_sub_2d, n_limbs: int = 8) -> list:
    """Parse the generator's designed body grid into per-env (effector_types, cap_types) specs.
    All torch-free (numpy / nested lists):
      counts_2d  (N, n_limbs)            effectors per limb, 0 = limb absent
      eff_sub_3d (N, n_limbs, max_len)   effector subtype per depth (only [:count] is read)
      cap_sub_2d (N, n_limbs)            cap subtype per limb (-1 where the limb was never capped)
    A limb with count 0 is absent regardless of its cap, so `bare cap at depth 0` == no limb."""
    out = []
    for e in range(len(counts_2d)):
        eff, caps = {}, {}
        for j in range(n_limbs):
            k = int(counts_2d[e][j])
            if k <= 0:
                continue
            assert k <= MAX_EFFECTORS, f"effector count {k} > MAX_EFFECTORS={MAX_EFFECTORS}"
            eff[j + 1] = [int(eff_sub_3d[e][j][d]) for d in range(k)]
            c = int(cap_sub_2d[e][j])
            caps[j + 1] = CANON_CAP if c < 0 else c
        assert eff, "0-module body; generator must guarantee >=1 effector"
        out.append((eff, caps))
    return out
