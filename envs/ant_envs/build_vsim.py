"""Programmatic vsim XML generation for 8-leg ant morphology variants.

A morphology is a set of active legs plus, per leg, a hip- and ankle-segment length. Every per-leg
geometric quantity is `magnitude * u`, where `u = (cos th, sin th, 0)` is the leg's outward unit
direction (th = (n-1)*45 deg). The magnitudes are functions of the two segment lengths:

  hip length H    : torso-attach -> ankle-joint distance. Sets the aux link + ankle-joint origin.
  ankle length A  : ankle-joint -> foot/contact distance.  Sets the leg link + force-sensor offset.

The hip-joint origin and torso stub are fixed (they belong to the torso, not a leg segment). Default
H/A reproduce the original fixed-geometry ant_8leg.vsim (numerically, not byte-for-byte).
"""
import math
from dataclasses import dataclass
from pathlib import Path

_GENERATED_DIR = Path(__file__).parent / "assets" / "generated"

_EFFORT  = "3.40282347e+38"
_DENSITY = "5.0"
_RADIUS  = "0.08"

# Segment-length defaults (match ant_8leg.vsim) and valid ranges (0.5x .. 1.5x default).
DEFAULT_HIP   = 0.282842712474619062   # torso-attach -> ankle joint
DEFAULT_ANKLE = 0.632455532033675882   # ankle joint -> foot/contact
HIP_RANGE     = (0.5 * DEFAULT_HIP,   1.5 * DEFAULT_HIP)     # (0.1414, 0.4243)
ANKLE_RANGE   = (0.5 * DEFAULT_ANKLE, 1.5 * DEFAULT_ANKLE)   # (0.3163, 0.9488)

_TORSO_RADIUS = 0.282842712474619062   # fixed hip-joint attach radius (= torso size)
_TORSO_STUB   = 0.141421356237309531   # fixed torso stub cylinder half-length & center
_LEG_CYL_FRAC = _TORSO_STUB * 2 / DEFAULT_ANKLE  # leg-cyl half-length/center per unit A (= 1/sqrt5)

# Outward unit direction per leg (exact, so axis-aligned legs stay clean).
_S = math.sqrt(0.5)
_DIR = {
    1: (1.0, 0.0, 0.0),  2: (_S,  _S, 0.0),  3: (0.0, 1.0, 0.0),  4: (-_S,  _S, 0.0),
    5: (-1.0, 0.0, 0.0), 6: (-_S, -_S, 0.0), 7: (0.0, -1.0, 0.0), 8: (_S,  -_S, 0.0),
}
# Cylinder rotation quat per leg (rotate th about z, w >= 0). Direction-only, so kept verbatim.
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
# Foot/contact (force-sensor) base offset per leg, at default ankle length. Kept verbatim from
# ant_8leg.vsim because the diagonal legs carry a ~0.04% rounding quirk there; scaled by A/A_default.
_D632 = 0.632455532033675882
_D447 = 0.447486996650695801
_SENSOR_BASE = {
    1: (_D632, 0.0, 0.0),   2: (_D447, _D447, 0.0),  3: (0.0, _D632, 0.0),  4: (-_D447, _D447, 0.0),
    5: (-_D632, 0.0, 0.0),  6: (-_D447, -_D447, 0.0), 7: (0.0, -_D632, 0.0), 8: (_D447, -_D447, 0.0),
}
# Ankle joint axis + limits per leg (direction-dependent, not a function of th alone).
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
    """One ant body: active legs plus a hip- and ankle-segment length per active leg."""
    legs: frozenset            # active leg indices 1..8
    hip_lengths: dict          # leg -> hip segment length
    ankle_lengths: dict        # leg -> ankle segment length

    @classmethod
    def from_legs(cls, legs) -> "Morphology":
        """Default-length body for a bare leg set (reproduces the original fixed geometry)."""
        legs = frozenset(legs)
        return cls(legs, {n: DEFAULT_HIP for n in legs}, {n: DEFAULT_ANKLE for n in legs})


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


def _aux_link(n: int, H: float) -> str:
    u, rot = _DIR[n], _CYL_ROT[n]
    return f"""    <link name="aux_{n}">{_INERTIAL_BLOCK}
{_cyl(H / 2, H / 2, u, rot)}
    </link>"""


def _leg_link(n: int, A: float) -> str:
    u, rot = _DIR[n], _CYL_ROT[n]
    return f"""    <link name="leg_{n}">{_INERTIAL_BLOCK}
{_cyl(_LEG_CYL_FRAC * A, _LEG_CYL_FRAC * A, u, rot)}
    </link>"""


def _hip_joint(n: int) -> str:
    o = _vec(_TORSO_RADIUS, _DIR[n])
    lo, hi = _HIP_LIMITS
    return f"""    <joint name="hip_{n}" type="revolute">
        <child link="aux_{n}"/>
        <parent link="torso"/>
        <origin xyz="{o}" rot="0.0 0.0 0.0 1.0"/>
        <linkTransform xyz="{o}" rot="0.0 0.0 0.0 1.0"/>
        <axis xyz="0.0 0.0 1.0"/>
        <limit lower="{lo}" upper="{hi}" effort="{_EFFORT}" velocity="{_JOINT_VELOCITY}"/>
        <dynamics {_JOINT_DYNAMICS}/>
    </joint>"""


def _ankle_joint(n: int, H: float) -> str:
    o = _vec(H, _DIR[n])
    ax = _ANKLE_AXIS[n]
    lo, hi = _ANKLE_LIMITS[n]
    return f"""    <joint name="ankle_{n}" type="revolute">
        <child link="leg_{n}"/>
        <parent link="aux_{n}"/>
        <origin xyz="{o}" rot="0.0 0.0 0.0 1.0"/>
        <linkTransform xyz="{o}" rot="0.0 0.0 0.0 1.0"/>
        <axis xyz="{ax}"/>
        <limit lower="{lo}" upper="{hi}" effort="{_EFFORT}" velocity="{_JOINT_VELOCITY}"/>
        <dynamics {_JOINT_DYNAMICS}/>
    </joint>"""


def _motor(n: int) -> str:
    return (
        f'        <motor name="hip_{n}" joint="hip_{n}" gear="150.0" lowLimit="-1.0" highLimit="1.0"/>\n'
        f'        <motor name="ankle_{n}" joint="ankle_{n}" gear="150.0" lowLimit="-1.0" highLimit="1.0"/>'
    )


def _sensor(n: int, A: float) -> str:
    r = A / DEFAULT_ANKLE
    b = _SENSOR_BASE[n]
    off = f"{_f(r * b[0])} {_f(r * b[1])} {_f(r * b[2])}"
    return f'        <sensor name="leg_{n}_sensor" link="leg_{n}" offset="{off}" flags="contact "/>'


def build_ant_vsim(morph: Morphology) -> str:
    """Generate vsim XML for a Morphology (active legs + per-leg hip/ankle lengths)."""
    active = sorted(morph.legs)
    H = morph.hip_lengths
    A = morph.ankle_lengths
    parts = ['<robot name="torso">']

    # Torso link: sphere + a fixed stub cylinder per active leg.
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

    for n in active:
        parts.append(_aux_link(n, H[n]))
    for n in active:
        parts.append(_leg_link(n, A[n]))

    # Hip/ankle joints declared in REVERSE active order so vsim's reverse-DFS traversal yields DOFs
    # in ascending active order (matches the scatter logic in AntMultiMorphEnv).
    for n in reversed(active):
        parts.append(_hip_joint(n))
    for n in reversed(active):
        parts.append(_ankle_joint(n, H[n]))

    parts.append('    <actuator>')
    for n in active:
        parts.append(_motor(n))
    parts.append('    </actuator>')

    if active:
        parts.append('    <forceSensor>')
        for n in active:
            parts.append(_sensor(n, A[n]))
        parts.append('    </forceSensor>')

    parts.append('</robot>')
    return '\n'.join(parts)


def write_vsim(morph: Morphology, index: int) -> Path:
    """Write a morphology's vsim XML to a deterministic per-index path (no caching; always rewrites)."""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _GENERATED_DIR / f"ant_morph_{index}.vsim"
    path.write_text(build_ant_vsim(morph))
    return path
