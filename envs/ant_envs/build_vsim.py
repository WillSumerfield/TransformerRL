"""Programmatic vsim XML generation for 8-leg ant morphology variants."""
import tempfile
from pathlib import Path

_EFFORT = "3.40282347e+38"
_DENSITY = "5.0"

# Per-leg geometry data extracted from ant_8leg.vsim
# Index 0 = leg 1, index 7 = leg 8
_LEG_DATA = [
    {   # Leg 1 (0 deg)
        "torso_cyl":  ("0.141421356237309531 0.0 0.0",  "0.0 0.0 0.0 1.0"),
        "aux_cyl":    ("0.141421356237309531 0.0 0.0",  "0.0 0.0 0.0 1.0"),
        "leg_cyl":    ("0.282842712474619062 0.0 0.0",  "0.0 0.0 0.0 1.0"),
        "hip_origin": "0.282842712474619062 0.0 0.0",
        "ankle_origin": "0.282842712474619062 0.0 0.0",
        "ankle_axis": "0.0 1.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "0.632455532033675882 0.0 0.0",
    },
    {   # Leg 2 (45 deg)
        "torso_cyl":  ("0.1 0.1 0.0",  "0.0 0.0 0.382683432365089782 0.923880"),
        "aux_cyl":    ("0.1 0.1 0.0",  "0.0 0.0 0.382683432365089782 0.923880"),
        "leg_cyl":    ("0.2 0.2 0.0",  "0.0 0.0 0.382683432365089782 0.923880"),
        "hip_origin": "0.2 0.2 0.0",
        "ankle_origin": "0.2 0.2 0.0",
        "ankle_axis": "-1.0 1.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "0.447486996650695801 0.447486996650695801 0.0",
    },
    {   # Leg 3 (90 deg)
        "torso_cyl":  ("0.0 0.141421356237309531 0.0",  "0.0 0.0 0.707106781186547573 0.707106781186547573"),
        "aux_cyl":    ("0.0 0.141421356237309531 0.0",  "0.0 0.0 0.707106781186547573 0.707106781186547573"),
        "leg_cyl":    ("0.0 0.282842712474619062 0.0",  "0.0 0.0 0.707106781186547573 0.707106781186547573"),
        "hip_origin": "0.0 0.282842712474619062 0.0",
        "ankle_origin": "0.0 0.282842712474619062 0.0",
        "ankle_axis": "-1.0 0.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "0.0 0.632455532033675882 0.0",
    },
    {   # Leg 4 (135 deg)
        "torso_cyl":  ("-0.1 0.1 0.0",  "0.0 0.0 0.923879532511286738 0.382683"),
        "aux_cyl":    ("-0.1 0.1 0.0",  "0.0 0.0 0.923879532511286738 0.382683"),
        "leg_cyl":    ("-0.2 0.2 0.0",  "0.0 0.0 0.923879532511286738 0.382683"),
        "hip_origin": "-0.2 0.2 0.0",
        "ankle_origin": "-0.2 0.2 0.0",
        "ankle_axis": "1.0 1.0 0.0",
        "ankle_limits": ("-1.745329300562540764", "-0.523598790168762207"),
        "sensor_offset": "-0.447486996650695801 0.447486996650695801 0.0",
    },
    {   # Leg 5 (180 deg)
        "torso_cyl":  ("-0.141421356237309531 0.0 0.0",  "0.0 0.0 1.0 0.0"),
        "aux_cyl":    ("-0.141421356237309531 0.0 0.0",  "0.0 0.0 1.0 0.0"),
        "leg_cyl":    ("-0.282842712474619062 0.0 0.0",  "0.0 0.0 1.0 0.0"),
        "hip_origin": "-0.282842712474619062 0.0 0.0",
        "ankle_origin": "-0.282842712474619062 0.0 0.0",
        "ankle_axis": "0.0 -1.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "-0.632455532033675882 0.0 0.0",
    },
    {   # Leg 6 (225 deg)
        "torso_cyl":  ("-0.1 -0.1 0.0",  "0.0 0.0 -0.923879532511286738 0.382683"),
        "aux_cyl":    ("-0.1 -0.1 0.0",  "0.0 0.0 -0.923879532511286738 0.382683"),
        "leg_cyl":    ("-0.2 -0.2 0.0",  "0.0 0.0 -0.923879532511286738 0.382683"),
        "hip_origin": "-0.2 -0.2 0.0",
        "ankle_origin": "-0.2 -0.2 0.0",
        "ankle_axis": "-1.0 1.0 0.0",
        "ankle_limits": ("-1.745329300562540764", "-0.523598790168762207"),
        "sensor_offset": "-0.447486996650695801 -0.447486996650695801 0.0",
    },
    {   # Leg 7 (270 deg)
        "torso_cyl":  ("0.0 -0.141421356237309531 0.0",  "0.0 0.0 -0.707106781186547573 0.707106781186547573"),
        "aux_cyl":    ("0.0 -0.141421356237309531 0.0",  "0.0 0.0 -0.707106781186547573 0.707106781186547573"),
        "leg_cyl":    ("0.0 -0.282842712474619062 0.0",  "0.0 0.0 -0.707106781186547573 0.707106781186547573"),
        "hip_origin": "0.0 -0.282842712474619062 0.0",
        "ankle_origin": "0.0 -0.282842712474619062 0.0",
        "ankle_axis": "1.0 0.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "0.0 -0.632455532033675882 0.0",
    },
    {   # Leg 8 (315 deg)
        "torso_cyl":  ("0.1 -0.1 0.0",  "0.0 0.0 -0.382683432365089782 0.923880"),
        "aux_cyl":    ("0.1 -0.1 0.0",  "0.0 0.0 -0.382683432365089782 0.923880"),
        "leg_cyl":    ("0.2 -0.2 0.0",  "0.0 0.0 -0.382683432365089782 0.923880"),
        "hip_origin": "0.2 -0.2 0.0",
        "ankle_origin": "0.2 -0.2 0.0",
        "ankle_axis": "1.0 1.0 0.0",
        "ankle_limits": ("0.523598790168762207", "1.745329300562540764"),
        "sensor_offset": "0.447486996650695801 -0.447486996650695801 0.0",
    },
]

_INERTIAL_BLOCK = f"""
        <inertial>
            <origin xyz="0.0 0.0 0.0" rot="0.0 0.0 0.0 1.0"/>
            <mass value="-1.0"/>
            <density value="{_DENSITY}"/>
            <inertia ixx="-1.0" iyy="-1.0" izz="-1.0" ixy="-1.0" ixz="-1.0" iyz="-1.0"/>
        </inertial>"""

_JOINT_DYNAMICS = 'damping="1.0" stiffness="100" friction="0.0" armature="1.0"'
_JOINT_VELOCITY = "30.0"
_HIP_LIMITS = ("-0.698131720225016239", "0.698131720225016239")


def _torso_cyl(d: dict) -> str:
    xyz, rot = d["torso_cyl"]
    return f"""        <collision>
            <origin xyz="{xyz}" rot="{rot}"/>
            <geometry>
                <cylinder length="0.141421356237309531" radius="0.08"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>"""


def _aux_link(n: int, d: dict) -> str:
    xyz, rot = d["aux_cyl"]
    return f"""    <link name="aux_{n}">{_INERTIAL_BLOCK}
        <collision>
            <origin xyz="{xyz}" rot="{rot}"/>
            <geometry>
                <cylinder length="0.141421356237309531" radius="0.08"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>
    </link>"""


def _leg_link(n: int, d: dict) -> str:
    xyz, rot = d["leg_cyl"]
    return f"""    <link name="leg_{n}">{_INERTIAL_BLOCK}
        <collision>
            <origin xyz="{xyz}" rot="{rot}"/>
            <geometry>
                <cylinder length="0.282842712474619062" radius="0.08"/>
                <queryProperties useQueries="true"/>
            </geometry>
        </collision>
    </link>"""


def _hip_joint(n: int, d: dict) -> str:
    o = d["hip_origin"]
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


def _ankle_joint(n: int, d: dict) -> str:
    o = d["ankle_origin"]
    ax = d["ankle_axis"]
    lo, hi = d["ankle_limits"]
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


def _sensor(n: int, d: dict) -> str:
    off = d["sensor_offset"]
    return f'        <sensor name="leg_{n}_sensor" link="leg_{n}" offset="{off}" flags="contact "/>'


def build_ant_vsim(active_legs: frozenset) -> str:
    """Generate vsim XML for the given set of active leg indices (1..8)."""
    active = sorted(active_legs)
    parts = ['<robot name="torso">']

    # Torso link: sphere + cylinders for active legs only
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
        parts.append(_torso_cyl(_LEG_DATA[n - 1]))
    parts.append('    </link>')

    # Aux links
    for n in active:
        parts.append(_aux_link(n, _LEG_DATA[n - 1]))

    # Leg links
    for n in active:
        parts.append(_leg_link(n, _LEG_DATA[n - 1]))

    # Hip joints — declared in REVERSE active order so vsim's reverse-DFS
    # traversal yields DOFs in ascending active order (matches scatter logic
    # in AntMultiMorphEnv.compute_observations).
    for n in reversed(active):
        parts.append(_hip_joint(n, _LEG_DATA[n - 1]))

    # Ankle joints
    for n in reversed(active):
        parts.append(_ankle_joint(n, _LEG_DATA[n - 1]))

    # Motors
    parts.append('    <actuator>')
    for n in active:
        parts.append(_motor(n))
    parts.append('    </actuator>')

    # Sensors
    if active:
        parts.append('    <forceSensor>')
        for n in active:
            parts.append(_sensor(n, _LEG_DATA[n - 1]))
        parts.append('    </forceSensor>')

    parts.append('</robot>')
    return '\n'.join(parts)


def write_vsim_tempfile(xml: str) -> Path:
    """Write vsim XML to a named temp file. Caller is responsible for deletion."""
    f = tempfile.NamedTemporaryFile(
        mode='w', suffix='.vsim', delete=False, prefix='ant_morph_'
    )
    f.write(xml)
    f.close()
    return Path(f.name)
