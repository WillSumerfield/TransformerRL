"""vsim body builder: raw XML-assembly + geometry-math primitives only. No knowledge of limbs,
rings, caps, torsos, or any other body-plan concept -- callers (a ModuleLibrary's construct_robot)
resolve every name/origin/axis to an absolute value and hand it to these emitters."""
import math
from pathlib import Path

_GENERATED_DIR = Path(__file__).parent / "assets" / "generated"


def fmt_float(v: float) -> str:
    return repr(0.0 if v == 0 else float(v))


def vec_str(mag: float, u: tuple) -> str:
    return f"{fmt_float(mag * u[0])} {fmt_float(mag * u[1])} {fmt_float(mag * u[2])}"


def quat_axis(axis: tuple, angle: float) -> tuple:
    s = math.sin(0.5 * angle)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle))


def quat_mul(a: tuple, b: tuple) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_str(q: tuple) -> str:
    return f"{fmt_float(q[0])} {fmt_float(q[1])} {fmt_float(q[2])} {fmt_float(q[3])}"


def quat_rotate(q: tuple, u: tuple) -> tuple:
    x, y, z, w = q
    t = (2 * (y * u[2] - z * u[1]), 2 * (z * u[0] - x * u[2]), 2 * (x * u[1] - y * u[0]))
    c = (y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0])
    return tuple(u[i] + w * t[i] + c[i] for i in range(3))


def geom_volume(kind: str, dims: tuple) -> float:
    if kind == "box":
        return 8.0 * dims[0] * dims[1] * dims[2]
    if kind == "cylinder":
        return math.pi * dims[1] ** 2 * 2 * dims[0]
    if kind == "sphere":
        return 4.0 / 3.0 * math.pi * dims[0] ** 3
    raise ValueError(f"unknown geom kind {kind}")


def geom_xml(kind: str, dims: tuple) -> str:
    if kind == "box":
        return f'<box size="{fmt_float(dims[0])} {fmt_float(dims[1])} {fmt_float(dims[2])}"/>'
    if kind == "cylinder":
        return f'<cylinder length="{fmt_float(dims[0])}" radius="{fmt_float(dims[1])}"/>'
    if kind == "sphere":
        return f'<sphere radius="{fmt_float(dims[0])}"/>'
    raise ValueError(f"unknown geom kind {kind}")


def collision_xml(origin: str, rot: str, kind: str, dims: tuple) -> str:
    return f"""        <collision>
            <origin xyz="{origin}" rot="{rot}"/>
            <geometry>
                {geom_xml(kind, dims)}
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>"""


def inertial_xml(density) -> str:
    """mass=-1 => vsim derives mass and inertia from the collision volume at this density."""
    return f"""
        <inertial>
            <origin xyz="0.0 0.0 0.0" rot="0.0 0.0 0.0 1.0"/>
            <mass value="-1.0"/>
            <density value="{density}"/>
            <inertia ixx="-1.0" iyy="-1.0" izz="-1.0" ixy="-1.0" ixz="-1.0" iyz="-1.0"/>
        </inertial>"""


def link_xml(name: str, density, collisions: list) -> str:
    body = "\n" + "\n".join(collisions) if collisions else ""
    return f'    <link name="{name}">{inertial_xml(density)}{body}\n    </link>'


def joint_xml(name: str, joint_type: str, parent: str, child: str, origin: str,
              rot: str = "0.0 0.0 0.0 1.0", axis: str = None, limits: tuple = None,
              effort=None, velocity=None, dynamics: str = None) -> str:
    """joint_type: revolute/prismatic/fixed. axis+limits are omitted (no DOF) for fixed joints."""
    parts = [f'    <joint name="{name}" type="{joint_type}">',
              f'        <child link="{child}"/>',
              f'        <parent link="{parent}"/>',
              f'        <origin xyz="{origin}" rot="{rot}"/>',
              f'        <linkTransform xyz="{origin}" rot="{rot}"/>']
    if axis is not None:
        parts.append(f'        <axis xyz="{axis}"/>')
    if limits is not None:
        lo, hi = limits
        parts.append(f'        <limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="{velocity}"/>')
    if dynamics is not None:
        parts.append(f'        <dynamics {dynamics}/>')
    parts.append('    </joint>')
    return "\n".join(parts)


def motor_xml(name: str, joint_name: str, gear, lo, hi) -> str:
    return f'        <motor name="{name}" joint="{joint_name}" gear="{gear}" lowLimit="{lo}" highLimit="{hi}"/>'


def sensor_xml(name: str, link: str, offset: str, flags: str = "contact ") -> str:
    return f'        <sensor name="{name}" link="{link}" offset="{offset}" flags="{flags}"/>'


def write_vsim(xml: str, index: int) -> Path:
    """Write an already-assembled vsim XML string to a deterministic per-index path (no caching)."""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _GENERATED_DIR / f"ant_morph_{index}.vsim"
    path.write_text(xml)
    return path
