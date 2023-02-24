import numpy as np
import math
import yaml


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


def retarget(insertHand):
    if insertHand.shape[0] != 63:
        return None
    angletotal2 = np.zeros(23)

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

    palm[0] = insertHand[0]
    palm[1] = insertHand[1]
    palm[2] = insertHand[2]
    ThumbEE[0] = insertHand[12]
    ThumbEE[1] = insertHand[13]
    ThumbEE[2] = insertHand[14]
    ThumbJ1[0] = insertHand[9]
    ThumbJ1[1] = insertHand[10]
    ThumbJ1[2] = insertHand[11]
    ThumbJ2[0] = insertHand[6]
    ThumbJ2[1] = insertHand[7]
    ThumbJ2[2] = insertHand[8]
    ThumbJ3[0] = insertHand[3]
    ThumbJ3[1] = insertHand[4]
    ThumbJ3[2] = insertHand[5]

    middleFingerEE[0] = insertHand[36]
    middleFingerEE[1] = insertHand[37]
    middleFingerEE[2] = insertHand[38]
    middleFingerJ1[0] = insertHand[33]
    middleFingerJ1[1] = insertHand[34]
    middleFingerJ1[2] = insertHand[35]
    middleFingerJ2[0] = insertHand[30]
    middleFingerJ2[1] = insertHand[31]
    middleFingerJ2[2] = insertHand[32]
    middleFingerJ3[0] = insertHand[27]
    middleFingerJ3[1] = insertHand[28]
    middleFingerJ3[2] = insertHand[29]

    indexFingerEE[0] = insertHand[24]
    indexFingerEE[1] = insertHand[25]
    indexFingerEE[2] = insertHand[26]
    indexFingerJ1[0] = insertHand[21]
    indexFingerJ1[1] = insertHand[22]
    indexFingerJ1[2] = insertHand[23]
    indexFingerJ2[0] = insertHand[18]
    indexFingerJ2[1] = insertHand[19]
    indexFingerJ2[2] = insertHand[20]
    indexFingerJ3[0] = insertHand[15]
    indexFingerJ3[1] = insertHand[16]
    indexFingerJ3[2] = insertHand[17]

    ringFingerEE[0] = insertHand[48]
    ringFingerEE[1] = insertHand[49]
    ringFingerEE[2] = insertHand[50]
    ringFingerJ1[0] = insertHand[45]
    ringFingerJ1[1] = insertHand[46]
    ringFingerJ1[2] = insertHand[47]
    ringFingerJ2[0] = insertHand[42]
    ringFingerJ2[1] = insertHand[43]
    ringFingerJ2[2] = insertHand[44]
    ringFingerJ3[0] = insertHand[39]
    ringFingerJ3[1] = insertHand[40]
    ringFingerJ3[2] = insertHand[41]

    PinkyFingerEE[0] = insertHand[60]
    PinkyFingerEE[1] = insertHand[61]
    PinkyFingerEE[2] = insertHand[62]
    PinkyFingerJ1[0] = insertHand[57]
    PinkyFingerJ1[1] = insertHand[58]
    PinkyFingerJ1[2] = insertHand[59]
    PinkyFingerJ2[0] = insertHand[54]
    PinkyFingerJ2[1] = insertHand[55]
    PinkyFingerJ2[2] = insertHand[56]
    PinkyFingerJ3[0] = insertHand[51]
    PinkyFingerJ3[1] = insertHand[52]
    PinkyFingerJ3[2] = insertHand[53]

    angletotal2 = calculateMiddleAngles(palm, ThumbEE, ThumbJ1, ThumbJ2, ThumbJ3, angletotal2, 1)
    angletotal2 = calculateMiddleAngles(palm, indexFingerEE, indexFingerJ1, indexFingerJ2, indexFingerJ3, angletotal2, 2)
    angletotal2 = calculateMiddleAngles(palm, middleFingerEE, middleFingerJ1, middleFingerJ2, middleFingerJ3, angletotal2, 3)
    angletotal2 = calculateMiddleAngles(palm, ringFingerEE, ringFingerJ1, ringFingerJ2, ringFingerJ3, angletotal2, 4)
    angletotal2 = calculateMiddleAngles(palm, PinkyFingerEE, PinkyFingerJ1, PinkyFingerJ2, PinkyFingerJ3, angletotal2, 5)

    b = np.zeros(16)
    b[0] = angletotal2[8]
    b[1] = angletotal2[7]
    b[2] = angletotal2[6]
    b[3] = angletotal2[5]
    b[4] = angletotal2[12]
    b[5] = angletotal2[11]
    b[6] = angletotal2[10]
    b[7] = angletotal2[9]
    b[8] = angletotal2[16]
    b[9] = angletotal2[15]
    b[10] = angletotal2[14]
    b[11] = angletotal2[13]
    b[12] = angletotal2[3]
    b[13] = angletotal2[2]
    b[14] = angletotal2[1]
    b[15] = angletotal2[0]

    # hard-code cnfoig
    joint_scale = np.ones(16)
    joint_offset = np.zeros(16)
    joint_mins = -np.ones(16)
    joint_maxs = np.ones(16)

    for i in range(16):
        if (i == 0) or (i == 4) or (i == 8) or (i == 12):
            b[i] = b[i] - 180
        else:
            b[i] = 180 - b[i]

    for i in range(16):
        b[i] = (b[i] * joint_scale[i]) + (b[i] * joint_offset[i])
        if (i == 0) or (i == 4) or (i == 8) or (i == 12):
            if b[i] > 10:
                b[i] = 10
            elif b[i] < -10:
                b[i] = -10
            else:
                b[i] = b[i] + 180
        else:
            if b[i] > 90:
                b[i] = 90
            b[i] = -b[i] + 180

    for i in range(16):
        if (i == 0) or (i == 4) or (i == 8) or (i == 12):
            b[i] = (abs((b[i] - 170) / 20) * (joint_maxs[i]) - joint_mins[i]) + joint_mins[i]
        else:
            b[i] = (((180 - b[i]) / 90) * (joint_maxs[i]) - joint_mins[i]) + joint_mins[i]
    return b
