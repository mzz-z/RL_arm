"""MuJoCo model validation for the 3-DOF arm environment."""

import numpy as np
import mujoco


def validate_model(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """
    Validate MuJoCo model configuration before training.

    Checks for required bodies, joints, actuators, sites, and constraints.
    Returns a dict with IDs for quick access during simulation.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        Dictionary with model element IDs

    Raises:
        AssertionError: If validation fails
    """
    ids = {}

    # Check joint count
    assert model.nq >= 3, f"Expected at least 3 joint positions, got {model.nq}"
    assert model.nv >= 3, f"Expected at least 3 joint velocities, got {model.nv}"
    assert model.nu >= 3, f"Expected at least 3 actuators, got {model.nu}"

    # Verify base joint exists
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base")
    assert base_id >= 0, "Missing 'base' joint in model"
    ids["base_joint"] = base_id

    # Verify shoulder joint exists
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
    assert shoulder_id >= 0, "Missing 'shoulder' joint in model"
    ids["shoulder_joint"] = shoulder_id

    # Verify elbow joint exists
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
    assert elbow_id >= 0, "Missing 'elbow' joint in model"
    ids["elbow_joint"] = elbow_id

    # Verify ee_site exists
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    assert ee_site_id >= 0, "Missing 'ee_site' in model"
    ids["ee_site"] = ee_site_id

    # Verify ball body exists
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    assert ball_id >= 0, "Missing 'ball' body in model"
    ids["ball_body"] = ball_id

    # Verify link2 body exists (for weld attachment)
    link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    assert link2_id >= 0, "Missing 'link2' body in model"
    ids["link2_body"] = link2_id

    # Verify weld constraint exists
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    assert weld_id >= 0, "Missing 'grasp_weld' equality constraint"
    ids["grasp_weld"] = weld_id

    # Verify actuators
    base_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_act")
    assert base_act_id >= 0, "Missing 'base_act' actuator"
    ids["base_act"] = base_act_id

    shoulder_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_act")
    assert shoulder_act_id >= 0, "Missing 'shoulder_act' actuator"
    ids["shoulder_act"] = shoulder_act_id

    elbow_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_act")
    assert elbow_act_id >= 0, "Missing 'elbow_act' actuator"
    ids["elbow_act"] = elbow_act_id

    # Verify actuator ranges match joint limits (approximately)
    base_jnt_range = model.jnt_range[base_id]
    base_act_range = model.actuator_ctrlrange[base_act_id]
    assert np.allclose(base_jnt_range, base_act_range, atol=0.1), (
        f"Base actuator range {base_act_range} doesn't match "
        f"joint range {base_jnt_range}"
    )

    shoulder_jnt_range = model.jnt_range[shoulder_id]
    shoulder_act_range = model.actuator_ctrlrange[shoulder_act_id]
    assert np.allclose(shoulder_jnt_range, shoulder_act_range, atol=0.1), (
        f"Shoulder actuator range {shoulder_act_range} doesn't match "
        f"joint range {shoulder_jnt_range}"
    )

    elbow_jnt_range = model.jnt_range[elbow_id]
    elbow_act_range = model.actuator_ctrlrange[elbow_act_id]
    assert np.allclose(elbow_jnt_range, elbow_act_range, atol=0.1), (
        f"Elbow actuator range {elbow_act_range} doesn't match "
        f"joint range {elbow_jnt_range}"
    )

    # Store joint limits for controller
    ids["joint_limits"] = {
        "base": tuple(base_jnt_range),
        "shoulder": tuple(shoulder_jnt_range),
        "elbow": tuple(elbow_jnt_range),
    }

    # Test forward kinematics
    mujoco.mj_forward(model, data)
    ee_pos = data.site_xpos[ee_site_id].copy()
    assert not np.any(np.isnan(ee_pos)), "Forward kinematics produced NaN for ee_site"

    ball_pos = data.xpos[ball_id].copy()
    assert not np.any(np.isnan(ball_pos)), "Forward kinematics produced NaN for ball"

    # Verify weld constraint is initially inactive (eq_active0 is the initial state from XML)
    assert model.eq_active0[weld_id] == 0, "grasp_weld should be initially inactive"

    # Get ball freejoint address
    ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    assert ball_jnt_id >= 0, "Missing 'ball_joint' freejoint"
    ids["ball_joint"] = ball_jnt_id
    ids["ball_qpos_addr"] = model.jnt_qposadr[ball_jnt_id]
    ids["ball_qvel_addr"] = model.jnt_dofadr[ball_jnt_id]

    # Get arm joint qpos/qvel addresses
    ids["base_qpos_addr"] = model.jnt_qposadr[base_id]
    ids["base_qvel_addr"] = model.jnt_dofadr[base_id]
    ids["shoulder_qpos_addr"] = model.jnt_qposadr[shoulder_id]
    ids["elbow_qpos_addr"] = model.jnt_qposadr[elbow_id]
    ids["shoulder_qvel_addr"] = model.jnt_dofadr[shoulder_id]
    ids["elbow_qvel_addr"] = model.jnt_dofadr[elbow_id]

    print("Model validation passed")

    return ids


def get_arm_geometry(model: mujoco.MjModel) -> dict:
    """
    Extract arm geometry from model for reward normalization.

    Returns:
        Dictionary with link lengths and max reach
    """
    # Get link geom IDs (try both old and new naming conventions)
    link1_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link1_main")
    if link1_geom_id < 0:
        link1_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link1_geom")

    link2_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2_main")
    if link2_geom_id < 0:
        link2_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2_geom")

    # Link lengths based on geometry:
    # Link1: shoulder to elbow = 0.25m
    # Link2: elbow to gripper center (ee_site at 0.26) = 0.26m
    link1_length = 0.25  # Default from XML
    link2_length = 0.26  # Updated for new gripper design (ee_site at 0.26)

    # Max reach is to the gripper fingertips
    max_reach = link1_length + link2_length  # 0.51m

    # Get table height
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_height = 0.4  # Default from XML
    if table_body_id >= 0:
        table_pos = model.body_pos[table_body_id]
        table_height = table_pos[2]

    # Get ball radius
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_radius = 0.03  # Default from XML
    if ball_geom_id >= 0:
        ball_radius = model.geom_size[ball_geom_id][0]

    return {
        "link1_length": link1_length,
        "link2_length": link2_length,
        "max_reach": max_reach,
        "table_height": table_height,
        "ball_radius": ball_radius,
    }
