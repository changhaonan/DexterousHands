import numpy as np
import math
import yaml


def human_hardcode_config():
    # hard-code cnfoig
    joint_scale = np.ones(23)
    joint_offset = np.zeros(23)
    joint_mins = -np.ones(23) * 180
    joint_maxs = np.ones(23) * 180
    # Thumb
    # J3
    joint_mins[0] = 150
    joint_maxs[0] = 160
    # J2
    joint_mins[1] = 150
    joint_maxs[1] = 160
    # J1
    joint_mins[2] = 150
    joint_maxs[2] = 160
    # J0
    joint_mins[3] = -2
    joint_maxs[3] = 2

    # Index
    # J3
    joint_mins[5] = 120
    joint_maxs[5] = 170
    # J2
    joint_mins[6] = 120
    joint_maxs[6] = 170
    # J1
    joint_mins[7] = 120
    joint_maxs[7] = 170
    # J0
    joint_mins[8] = -2
    joint_maxs[8] = 2

    # Middle
    # J3
    joint_mins[9] = 120
    joint_maxs[9] = 170
    # J2
    joint_mins[10] = 120
    joint_maxs[10] = 170
    # J1
    joint_mins[11] = 120
    joint_maxs[11] = 170
    # J0
    joint_mins[12] = -2
    joint_maxs[12] = 2

    return joint_scale, joint_offset, joint_mins, joint_maxs


def robotiq_hardcode_config():
    # hard-code cnfoig
    joint_scale = -np.ones(16)
    joint_offset = np.zeros(16)
    joint_mins = -np.ones(16)
    joint_maxs = np.ones(16)

    # Thumb
    # J1
    joint_mins[0] = -1
    joint_maxs[0] = -0.3
    # J2
    joint_mins[1] = -1
    joint_maxs[1] = -0.3
    # J3
    joint_mins[2] = -1
    joint_maxs[2] = -0.3

    # Index
    # J0
    joint_mins[3] = -1
    joint_maxs[3] = -0.3
    # J1
    joint_mins[4] = -1
    joint_maxs[4] = -0.3
    # J2
    joint_mins[5] = -1
    joint_maxs[5] = -0.3
    # J3
    joint_mins[6] = -1
    joint_maxs[6] = -0.90

    # Middle
    # J0
    joint_mins[7] = -1
    joint_maxs[7] = -0.3
    # J1
    joint_mins[8] = -1
    joint_maxs[8] = -0.3
    # J2
    joint_mins[9] = -1
    joint_maxs[9] = -0.3
    # J3
    joint_mins[10] = -1
    joint_maxs[10] = -0.90

    return joint_scale, joint_offset, joint_mins, joint_maxs


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angleBetweenVectors(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    result = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return result * 180 / math.pi


def vectorFromPoints(p1, p2):
    return np.subtract(p1, p2)


def rotatebyDegrees(point, output, theta, input):
    theta = theta * math.pi / 180
    if input == "x_axis":
        output[0] = point[0]
        output[1] = point[1] * math.cos(theta) + point[2] * -math.sin(theta)
        output[2] = point[1] * math.sin(theta) + point[2] * math.cos(theta)
    elif input == "y_axis":
        output[0] = point[0] * math.cos(theta) + point[2] * math.sin(theta)
        output[1] = point[1]
        output[2] = point[0] * -math.sin(theta) + point[2] * math.cos(theta)
    else:
        output[0] = point[0] * math.cos(theta) - point[1] * math.sin(theta)
        output[1] = point[0] * math.sin(theta) + point[1] * math.cos(theta)
        output[2] = point[2]
    return output


def calculateMiddleAngles(palm_in, mf_ee_in, mf_j1_in, mf_j2_in, mf_j3_in, angleTotal, wfing):
    mf_ee = np.zeros(3)
    mf_j1 = np.zeros(3)
    mf_j2 = np.zeros(3)
    mf_j3 = np.zeros(3)
    palm = np.zeros(3)
    for i in range(3):
        mf_ee[i] = mf_ee_in[i] - palm_in[i]
        mf_j1[i] = mf_j1_in[i] - palm_in[i]
        mf_j2[i] = mf_j2_in[i] - palm_in[i]
        mf_j3[i] = mf_j3_in[i] - palm_in[i]
        palm[i] = palm_in[i] - palm_in[i]

    plane_point = [mf_j3[0], 0, 0]
    plane_point2 = [mf_j3[0], mf_j3[1], 0]
    curr_angle = angleBetweenVectors(plane_point, plane_point2)

    ee_r = np.zeros(3)
    j1_r = np.zeros(3)
    j2_r = np.zeros(3)
    j3_r = np.zeros(3)
    j3_r = rotatebyDegrees(mf_j3, j3_r, curr_angle, "z_axis")
    if (j3_r[1] > 0.00001) or (j3_r[1] < -0.00001):
        curr_angle = 180 - curr_angle
        j3_r = rotatebyDegrees(mf_j3, j3_r, curr_angle, "z_axis")
    ee_r = rotatebyDegrees(mf_ee, ee_r, curr_angle, "z_axis")
    j1_r = rotatebyDegrees(mf_j1, j1_r, curr_angle, "z_axis")
    j2_r = rotatebyDegrees(mf_j2, j2_r, curr_angle, "z_axis")

    plane_point3 = [j3_r[0], 0, 0]
    plane_point4 = [j3_r[0], 0, j3_r[2]]
    curr_angle2 = angleBetweenVectors(plane_point3, plane_point4)

    ee_r2 = np.zeros(3)
    j1_r2 = np.zeros(3)
    j2_r2 = np.zeros(3)
    j3_r2 = np.zeros(3)
    j3_r2 = rotatebyDegrees(j3_r, j3_r2, curr_angle2, "y_axis")
    if (j3_r2[2] > 0.00001) or (j3_r2[2] < -0.00001):
        curr_angle2 = 180 - curr_angle2
        j3_r2 = rotatebyDegrees(j3_r, j3_r2, curr_angle2, "y_axis")
    ee_r2 = rotatebyDegrees(ee_r, ee_r2, curr_angle2, "y_axis")
    j1_r2 = rotatebyDegrees(j1_r, j1_r2, curr_angle2, "y_axis")
    j2_r2 = rotatebyDegrees(j2_r, j2_r2, curr_angle2, "y_axis")

    plane_point5 = [0, 0, j2_r2[2]]
    plane_point6 = [0, j2_r2[1], j2_r2[2]]
    curr_angle3 = angleBetweenVectors(plane_point5, plane_point6)

    ee_r3 = np.zeros(3)
    j1_r3 = np.zeros(3)
    j2_r3 = np.zeros(3)
    j2_r3 = rotatebyDegrees(j2_r2, j2_r3, curr_angle3, "x_axis")
    if (j2_r3[1] > 0.00001) or (j2_r3[1] < -0.00001):
        curr_angle3 = 180 - curr_angle3
        j2_r3 = rotatebyDegrees(j2_r2, j2_r3, curr_angle3, "x_axis")
    ee_r3 = rotatebyDegrees(ee_r2, ee_r3, curr_angle3, "x_axis")
    j1_r3 = rotatebyDegrees(j1_r2, j1_r3, curr_angle3, "x_axis")

    if ee_r3[2] < 0:
        ee_r3[1] = -ee_r3[1]
        ee_r3[2] = -ee_r3[2]
        j1_r3[1] = -j1_r3[1]
        j1_r3[2] = -j1_r3[2]
        j2_r3[1] = -j2_r3[1]
        j2_r3[2] = -j2_r3[2]

    ee_j1 = np.zeros(3)
    j1_j2 = np.zeros(3)
    j2_j3 = np.zeros(3)
    j3_palm = np.zeros(3)
    ee_j1 = vectorFromPoints(j1_r3, ee_r3)
    j1_j2 = vectorFromPoints(j1_r3, j2_r3)
    j2_j3 = vectorFromPoints(j3_r2, j2_r3)
    j3_palm = vectorFromPoints(j3_r2, palm)

    theta_1 = angleBetweenVectors(ee_j1, j1_j2)
    theta_2 = angleBetweenVectors(j1_j2, j2_j3)

    temp2 = [0, 0, j1_j2[2]]
    temp3 = [0, j1_j2[1], j1_j2[2]]
    theta_4 = angleBetweenVectors(temp2, temp3)

    if j1_r3[1] > 0:
        theta_4 = 180 + theta_4
    else:
        theta_4 = 180 - theta_4

    theta_3 = angleBetweenVectors(j2_j3, j3_palm)

    theta_4 = 180 + ((theta_4 - 180) / 10)

    if wfing == 1:
        angleTotal[0] = theta_1
        angleTotal[1] = theta_2
        angleTotal[2] = theta_3
        angleTotal[3] = theta_4
    elif wfing == 2:
        angleTotal[5] = theta_1
        angleTotal[6] = theta_2
        angleTotal[7] = theta_3
        angleTotal[8] = theta_4
    elif wfing == 3:
        angleTotal[9] = theta_1
        angleTotal[10] = theta_2
        angleTotal[11] = theta_3
        angleTotal[12] = theta_4
    elif wfing == 4:
        angleTotal[13] = theta_1
        angleTotal[14] = theta_2
        angleTotal[15] = theta_3
        angleTotal[16] = theta_4
    elif wfing == 5:
        angleTotal[17] = theta_1
        angleTotal[18] = theta_2
        angleTotal[19] = theta_3
        angleTotal[20] = theta_4
    angleTotal[21] = 180
    angleTotal[22] = 180
    return angleTotal


def retarget(human_joint_3d, robot_type="robotiq"):
    if human_joint_3d.shape[0] != 63:
        return None
    human_joint = np.zeros(23)

    # assign palm
    palm = np.zeros(3)

    # assign ThumbEE ThumbJ1 ThumbJ2, ThumbJ3
    ThumbEE = np.zeros(3)
    ThumbJ1 = np.zeros(3)
    ThumbJ2 = np.zeros(3)
    ThumbJ3 = np.zeros(3)
    # assign middleFingerEE, middleFingerJ1, middleFingerJ2, middleFingerJ3
    middleFingerEE = np.zeros(3)
    middleFingerJ1 = np.zeros(3)
    middleFingerJ2 = np.zeros(3)
    middleFingerJ3 = np.zeros(3)
    # assign indexFingerEE, indexFingerJ1, indexFingerJ2, indexFingerJ3
    indexFingerEE = np.zeros(3)
    indexFingerJ1 = np.zeros(3)
    indexFingerJ2 = np.zeros(3)
    indexFingerJ3 = np.zeros(3)
    # assign ringFingerEE, ringFingerJ1, ringFingerJ2, ringFingerJ3
    ringFingerEE = np.zeros(3)
    ringFingerJ1 = np.zeros(3)
    ringFingerJ2 = np.zeros(3)
    ringFingerJ3 = np.zeros(3)
    # assign PinkyFingerEE, PinkyFingerJ1, PinkyFingerJ2, PinkyFingerJ3
    PinkyFingerEE = np.zeros(3)
    PinkyFingerJ1 = np.zeros(3)
    PinkyFingerJ2 = np.zeros(3)
    PinkyFingerJ3 = np.zeros(3)

    palm[0] = human_joint_3d[0]
    palm[1] = human_joint_3d[1]
    palm[2] = human_joint_3d[2]
    ThumbEE[0] = human_joint_3d[12]
    ThumbEE[1] = human_joint_3d[13]
    ThumbEE[2] = human_joint_3d[14]
    ThumbJ1[0] = human_joint_3d[9]
    ThumbJ1[1] = human_joint_3d[10]
    ThumbJ1[2] = human_joint_3d[11]
    ThumbJ2[0] = human_joint_3d[6]
    ThumbJ2[1] = human_joint_3d[7]
    ThumbJ2[2] = human_joint_3d[8]
    ThumbJ3[0] = human_joint_3d[3]
    ThumbJ3[1] = human_joint_3d[4]
    ThumbJ3[2] = human_joint_3d[5]

    middleFingerEE[0] = human_joint_3d[36]
    middleFingerEE[1] = human_joint_3d[37]
    middleFingerEE[2] = human_joint_3d[38]
    middleFingerJ1[0] = human_joint_3d[33]
    middleFingerJ1[1] = human_joint_3d[34]
    middleFingerJ1[2] = human_joint_3d[35]
    middleFingerJ2[0] = human_joint_3d[30]
    middleFingerJ2[1] = human_joint_3d[31]
    middleFingerJ2[2] = human_joint_3d[32]
    middleFingerJ3[0] = human_joint_3d[27]
    middleFingerJ3[1] = human_joint_3d[28]
    middleFingerJ3[2] = human_joint_3d[29]

    indexFingerEE[0] = human_joint_3d[24]
    indexFingerEE[1] = human_joint_3d[25]
    indexFingerEE[2] = human_joint_3d[26]
    indexFingerJ1[0] = human_joint_3d[21]
    indexFingerJ1[1] = human_joint_3d[22]
    indexFingerJ1[2] = human_joint_3d[23]
    indexFingerJ2[0] = human_joint_3d[18]
    indexFingerJ2[1] = human_joint_3d[19]
    indexFingerJ2[2] = human_joint_3d[20]
    indexFingerJ3[0] = human_joint_3d[15]
    indexFingerJ3[1] = human_joint_3d[16]
    indexFingerJ3[2] = human_joint_3d[17]

    ringFingerEE[0] = human_joint_3d[48]
    ringFingerEE[1] = human_joint_3d[49]
    ringFingerEE[2] = human_joint_3d[50]
    ringFingerJ1[0] = human_joint_3d[45]
    ringFingerJ1[1] = human_joint_3d[46]
    ringFingerJ1[2] = human_joint_3d[47]
    ringFingerJ2[0] = human_joint_3d[42]
    ringFingerJ2[1] = human_joint_3d[43]
    ringFingerJ2[2] = human_joint_3d[44]
    ringFingerJ3[0] = human_joint_3d[39]
    ringFingerJ3[1] = human_joint_3d[40]
    ringFingerJ3[2] = human_joint_3d[41]

    PinkyFingerEE[0] = human_joint_3d[60]
    PinkyFingerEE[1] = human_joint_3d[61]
    PinkyFingerEE[2] = human_joint_3d[62]
    PinkyFingerJ1[0] = human_joint_3d[57]
    PinkyFingerJ1[1] = human_joint_3d[58]
    PinkyFingerJ1[2] = human_joint_3d[59]
    PinkyFingerJ2[0] = human_joint_3d[54]
    PinkyFingerJ2[1] = human_joint_3d[55]
    PinkyFingerJ2[2] = human_joint_3d[56]
    PinkyFingerJ3[0] = human_joint_3d[51]
    PinkyFingerJ3[1] = human_joint_3d[52]
    PinkyFingerJ3[2] = human_joint_3d[53]

    # Calculate human joint angles
    human_joint = calculateMiddleAngles(palm, ThumbEE, ThumbJ1, ThumbJ2, ThumbJ3, human_joint, 1)
    human_joint = calculateMiddleAngles(palm, indexFingerEE, indexFingerJ1, indexFingerJ2, indexFingerJ3, human_joint, 2)
    human_joint = calculateMiddleAngles(palm, middleFingerEE, middleFingerJ1, middleFingerJ2, middleFingerJ3, human_joint, 3)
    human_joint = calculateMiddleAngles(palm, ringFingerEE, ringFingerJ1, ringFingerJ2, ringFingerJ3, human_joint, 4)
    human_joint = calculateMiddleAngles(palm, PinkyFingerEE, PinkyFingerJ1, PinkyFingerJ2, PinkyFingerJ3, human_joint, 5)

    # Hardcode config
    joint_scale, joint_offset, joint_mins, joint_maxs = human_hardcode_config()

    # Normalize for rotation angle
    for i in range(16):
        if i == 8 or i == 12 or i == 3:
            human_joint[i] = 0

    # Scale it & clip it
    human_joint_clip = np.clip(human_joint * joint_scale + joint_offset, joint_mins, joint_maxs)
    # Normalize to -1 to 1
    human_joint_norm = (human_joint_clip - joint_mins) / (joint_maxs - joint_mins) * 2 - 1

    print("thumb: ", human_joint[0], ", norm: ", human_joint_norm[0])
    if robot_type == "robotiq":
        # Remapping to robot joint angles
        robot_joint = -np.ones(16)
        # Thumb: J3 is the most reponsing joint
        robot_joint[0] = human_joint_norm[5]
        robot_joint[1] = human_joint_norm[5]
        robot_joint[2] = human_joint_norm[5]
        # Index
        robot_joint[3] = human_joint_norm[5]
        robot_joint[4] = human_joint_norm[5]
        robot_joint[5] = human_joint_norm[5]
        robot_joint[6] = human_joint_norm[5]
        # Middle
        robot_joint[7] = human_joint_norm[5]
        robot_joint[8] = human_joint_norm[5]
        robot_joint[9] = human_joint_norm[5]
        robot_joint[10] = human_joint_norm[5]

        joint_scale, joint_offset, joint_mins, joint_maxs = robotiq_hardcode_config()
        robot_joint_clip = np.clip(robot_joint * joint_scale + joint_offset, joint_mins, joint_maxs)
        return robot_joint_clip
    else:
        raise ValueError("Unknown robot type!")
