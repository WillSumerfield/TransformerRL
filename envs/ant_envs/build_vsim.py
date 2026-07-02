"""PHASE-1 DRAFT of envs/ant_envs/build_vsim.py — variable-length limbs (module chains).

Not wired in. Drafted in scratchpad so the live tree (and the running tuner) is untouched.
Collapses hip_lengths/ankle_lengths -> module_lengths (ADR-0014 deferred data model) and
generalizes the fixed torso->aux->leg chain into torso->mod_1->mod_2->...->mod_k.

Design (grill 2026-07-02, recorded in temp/ComplexityEscalationPlan.md):
  - A limb = 1..MAX_LIMB_LENGTH actuated modules along the outward unit dir u=_DIR[n].
  - pos1 (proximal) = SWING joint: axis 0 0 1, hip limits, HIP_RANGE length  (== old hip/aux).
  - pos>=2 (distal)  = KNEE  joint: _ANKLE_AXIS[n], ankle limits, ANKLE_RANGE  (== old ankle/leg, repeated).
  - joint origin of module i = (torso_radius if i==1 else length[i-1]) * u.
  - foot/contact force-sensor on the TERMINAL module only (one contact per limb).
  - names: aux_n/leg_n -> mod_{n}_{i}; hip_n/ankle_n -> joint_{n}_{i}; "torso" root stays.
  - length-2 default body MUST reproduce the current ant exactly (geometry).

OPEN / defer to branch-time runtime check: the joint DECLARATION ORDER below assumes vsim's
reverse-DFS still yields a depth-major DOF index layout ((d-1)*8 + (n-1)). That coupling to the
scatter in ant_multimorph can only be confirmed by loading a built body in the sim. Flagged.
"""
import math
from dataclasses import dataclass
from pathlib import Path

_GENERATED_DIR = Path(__file__).parent / "assets" / "generated"

MAX_LIMB_LENGTH = 4

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
_JOINT_DYNAMICS = 'damping="1.0" stiffness="100" friction="0.0" armature="1.0"'
_JOINT_VELOCITY = "30.0"
_INERTIAL_BLOCK = f"""
        <inertial>
            <origin xyz="0.0 0.0 0.0" rot="0.0 0.0 0.0 1.0"/>
            <mass value="-1.0"/>
            <density value="{_DENSITY}"/>
            <inertia ixx="-1.0" iyy="-1.0" izz="-1.0" ixy="-1.0" ixz="-1.0" iyz="-1.0"/>
        </inertial>"""


@dataclass
class Morphology:
    """One robot body: per active limb, an ordered list of module lengths (proximal -> distal).

    module_lengths: {limb 1..8 -> [len_pos1, len_pos2, ...]}, list length = # modules (1..MAX).
    A limb is present iff it has >=1 module. Replaces the fixed hip_lengths/ankle_lengths dicts.
    """
    module_lengths: dict  # limb -> list[float]

    @property
    def legs(self) -> frozenset:
        return frozenset(n for n, L in self.module_lengths.items() if L)

    def num_modules(self, n: int) -> int:
        return len(self.module_lengths.get(n, ()))

    @classmethod
    def from_legs(cls, legs) -> "Morphology":
        """Default-length, length-2 body for a bare limb set (reproduces the current fixed ant)."""
        return cls({n: [DEFAULT_HIP, DEFAULT_ANKLE] for n in frozenset(legs)})

    @classmethod
    def from_counts(cls, counts: dict) -> "Morphology":
        """Default-length body from per-limb module counts {limb -> k (1..MAX)}. Each limb is a
        chain of k default-length modules: pos1 = HIP default (swing), pos2.. = ANKLE default (knee).
        This is what the codesign generator produces in Phase 1 — it designs module COUNT per limb;
        per-module continuous lengths stay deferred (fixed at defaults)."""
        return cls({n: [DEFAULT_HIP] + [DEFAULT_ANKLE] * (k - 1)
                    for n, k in counts.items() if k > 0})

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


def _cyl(center: float, half_len: float, u: tuple, rot: str) -> str:
    return f"""        <collision>
            <origin xyz="{_vec(center, u)}" rot="{rot}"/>
            <geometry>
                <cylinder length="{_f(half_len)}" radius="{_RADIUS}"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>"""


def _module_link(n: int, i: int, length: float) -> str:
    """Link for module i (1-based) of limb n. i==1 uses the proximal (aux/hip) cylinder convention
    (full-length cyl), i>=2 uses the distal (leg/ankle) convention (_LEG_CYL_FRAC shortened cyl)."""
    u, rot = _DIR[n], _CYL_ROT[n]
    if i == 1:
        cyl = _cyl(length / 2, length / 2, u, rot)          # == old aux_link(H)
    else:
        cyl = _cyl(_LEG_CYL_FRAC * length, _LEG_CYL_FRAC * length, u, rot)  # == old leg_link(A)
    return f"""    <link name="mod_{n}_{i}">{_INERTIAL_BLOCK}
{cyl}
    </link>"""


def _module_joint(n: int, i: int, prev_length: float) -> str:
    """Joint feeding module i of limb n. i==1: swing (z-axis, hip limits), origin at torso radius,
    parent torso. i>=2: knee (_ANKLE_AXIS[n], ankle limits), origin at prev module length, parent mod i-1."""
    if i == 1:
        parent = "torso"
        origin = _vec(_TORSO_RADIUS, _DIR[n])
        axis = "0.0 0.0 1.0"
        lo, hi = _HIP_LIMITS
    else:
        parent = f"mod_{n}_{i-1}"
        origin = _vec(prev_length, _DIR[n])
        axis = _ANKLE_AXIS[n]
        lo, hi = _ANKLE_LIMITS[n]
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


def _sensor(n: int, k: int, term_length: float) -> str:
    """Foot/contact sensor on the terminal module k of limb n, scaled by its length."""
    r = term_length / DEFAULT_ANKLE
    b = _SENSOR_BASE[n]
    off = f"{_f(r * b[0])} {_f(r * b[1])} {_f(r * b[2])}"
    return f'        <sensor name="mod_{n}_{k}_sensor" link="mod_{n}_{k}" offset="{off}" flags="contact "/>'


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

    # Links declared DEPTH-MAJOR, ascending active — matches the old "all aux, then all leg" order
    # so a length-2 body is byte-identical (modulo names) to the current ant.
    for d in range(1, maxd + 1):
        for n in active:
            if len(L[n]) >= d:
                parts.append(_module_link(n, d, L[n][d - 1]))

    # Joints declared DEPTH-MAJOR, reverse-active within each depth (extends the old
    # "reversed hips, then reversed ankles" so reverse-DFS yields depth-major ascending DOFs).
    # RUNTIME-VERIFY at branch time that this reproduces the length-2 DOF ordering.
    for d in range(1, maxd + 1):
        for n in reversed(active):
            if len(L[n]) >= d:
                prev_len = L[n][d - 2] if d >= 2 else 0.0
                parts.append(_module_joint(n, d, prev_len))

    # Motors declared per-limb (n ascending, then depth) — matches the old actuator order so the
    # length-2 body is byte-identical (modulo names) to the current ant. vsim maps motor->joint by
    # name, so XML order is not load-bearing, but we keep it identical to remove all doubt.
    parts.append('    <actuator>')
    for n in active:
        for i in range(1, len(L[n]) + 1):
            parts.append(_motor(n, i))
    parts.append('    </actuator>')

    if active:
        parts.append('    <forceSensor>')
        for n in active:
            k = len(L[n])
            parts.append(_sensor(n, k, L[n][k - 1]))
        parts.append('    </forceSensor>')

    parts.append('</robot>')
    return '\n'.join(parts)


def write_vsim(morph: Morphology, index: int) -> Path:
    """Write a morphology's vsim XML to a deterministic per-index path (no caching; always rewrites)."""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _GENERATED_DIR / f"ant_morph_{index}.vsim"
    path.write_text(build_ant_vsim(morph))
    return path


def bodies_from_counts(counts_2d, n_limbs: int = 8) -> list:
    """Parse a per-env module-count grid into per-env {limb -> count} specs (torch-free; accepts a
    numpy array or nested list, (N, n_limbs), values 0..MAX_LIMB_LENGTH). Presence is emergent
    (count 0 = limb absent). Enforces the >=1-module and range invariants the generator must uphold."""
    bodies = []
    for row in counts_2d:
        spec = {j + 1: int(row[j]) for j in range(n_limbs) if int(row[j]) > 0}
        assert spec, "0-module body; generator must guarantee >=1 module"
        assert all(1 <= k <= MAX_LIMB_LENGTH for k in spec.values()), \
            f"module count out of range 1..{MAX_LIMB_LENGTH}: {spec}"
        bodies.append(spec)
    return bodies
