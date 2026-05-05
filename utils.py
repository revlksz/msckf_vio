import numpy as np

def skew(v):
    """
    Returns the 3x3 skew-symmetric matrix of a 3D vector.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def quaternion_multiply(q, p):
    """
    Multiply two quaternions [w, x, y, z]
    q * p
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    Direction: Body to World (C(q)^T in some notation, but here we assume q_W transforms World to Body? No, standard is q_W is orientation of body in world, so C(q_W) is body to world)
    Actually, let's use the convention: C(q) is the rotation matrix from World to Body.
    Wait, the user's prompt says: "C(q) kuaterniyondan rotasyon matrisi", "v_dot = C(q) * (a_m - b_a) + g".
    If v is in world frame, and a_m is in body frame, then the rotation must be Body to World.
    Let's standardise: R_WB = Rotation from Body to World.
    C(q) typically denotes World to Body in aviation (R_BW).
    Let's stick to quaternion representing rotation from Body to World (R_WB), so q_W maps body vector to world vector.
    """
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    q = q / n
    w, x, y, z = q
    
    # R_WB
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def Omega(w):
    """
    Returns the 4x4 matrix for quaternion kinematics.
    """
    return np.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0]
    ])

def normalize_quaternion(q):
    return q / np.linalg.norm(q)
