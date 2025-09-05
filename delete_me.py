import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rot6d_rowmajor(quat_xyzw):
    """Your current way: lR[:, :2].reshape(-1) (row-major)."""
    lR = R.from_quat(quat_xyzw).as_matrix()
    return lR[:, :2].reshape(-1)  # row-major flatten

def quat_to_rot6d_colmajor(quat_xyzw):
    """Correct way: take first two columns, column-major flatten."""
    lR = R.from_quat(quat_xyzw).as_matrix()
    return lR[:, :2].reshape(-1, order='F')  # col-major flatten

# Example quaternion: 90° about Z axis (should rotate x->y, y->-x)
quat_xyzw = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # xyzw

row_major_6d = quat_to_rot6d_rowmajor(quat_xyzw)
col_major_6d = quat_to_rot6d_colmajor(quat_xyzw)

print("Rotation matrix:\n", R.from_quat(quat_xyzw).as_matrix())
print("\nRow-major flatten (your current code):\n", row_major_6d)
print("Col-major flatten (correct 6D convention):\n", col_major_6d)