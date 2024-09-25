import numpy as np

# 1.) VALIDATING ROTATIONS

def check_SOn(matrix: np.ndarray, epsilon: float = 0.01) -> bool:
    # See pgs. 68-70 of Modern Robotics textbook
    # Check dimensions of matrix
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] not in {2, 3}:
        return False

    # Check whether this statement is true: R^T*R=I, where R^T is the transpose of R
    identity = np.eye(matrix.shape[0])
    if not np.allclose(np.dot(matrix, matrix.T), identity, atol=epsilon):
        return False

    # Check whether determinant of R = 1
    if not np.isclose(np.linalg.det(matrix), 1, atol=epsilon):
        return False

    return True

def check_quaternion(vector: np.ndarray, epsilon: float = 0.01) -> bool:
    # See Appendix B of Modern Robotics textbook (pgs. 581-582)
    # Check whether the vector is a 4-vector
    if vector.shape != (4,) and vector.shape != (4, 1):
        return False

    # Check if the magnitude is 1
    magnitude = np.linalg.norm(vector)
    return np.isclose(magnitude, 1, atol=epsilon)

def check_SEn(matrix: np.ndarray, epsilon: float = 0.01) -> bool:
    # See pgs. 89-90 of Modern Robotics textbook
    # https://youtu.be/vlb3P7arbkU?si=kdhsJ0rCTVFhPP55
    # https://youtu.be/09I15RO49vg?si=3dyr9SLyAj1lEXPi

    # Check if n is 2 or 3 (must be (n+1)*(n+1)) and if matrix is square
    n = matrix.shape[0] - 1
    if matrix.shape[0] != matrix.shape[1] or n not in {2, 3}:
        return False

    # SE(n) contains a translation vector and SO(n), check SO(n)
    SOn_segment = matrix[:-1, :-1]
    if not check_SOn(SOn_segment, epsilon):
        return False

    # The last row is [0, ..., 0, 1] for homogeneity
    last = matrix[-1, :]
    test = np.zeros(n + 1)
    test[-1] = 1
    if not np.allclose(last, test, atol=epsilon):
        return False

    return True


# 1.1 extra:

def correct_SOn(matrix: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    # if matrix is SO(n) it does not need correcting
    if check_SOn(matrix, epsilon): return matrix

    # if matrix is not orthogonal, orthogonalize it (SVD)
    U, sig, Vt = np.linalg.svd(matrix)
    matrix = np.dot(U, Vt)
    det_matrix = np.linalg.det(matrix)

    # if determinant is -1, negate the last column to flip the determinant
    if det_matrix < 0:
        matrix[:,-1] = -matrix[:,-1]

    return matrix

def correct_quaternion(vector: np.ndarray, epsilon: float=0.01) -> np.ndarray:
    # if v is a quaternion, return v
    if check_quaternion(vector, epsilon=0.01): return vector

    # if v does not have length 1, normalize it
    norm = np.linalg.norm(vector)
    if norm != 1:
        vector = vector / norm

    return vector

def correct_SEn(matrix: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    # find out the size of the matrix
    n = matrix.shape[0] - 1

    # take the rotation matrix, and correct it to be orthogonal
    r = matrix[0:n, 0:n]
    r = correct_SOn(r)
    matrix[0:n, 0:n] = r

    # ensure the last row is correct, with 0s and 1s
    matrix[-1] = np.zeros(n+1)
    matrix[-1, -1] = 1

    return matrix

# Testing methods
# def test(method: int):
#     if method == 1:
#         matrix = np.random.rand(2,2)
#         # print(matrix)
#         # print(check_SOn(matrix))
#         matrix = correct_SOn(matrix)
#         # print(matrix)
#         # print(check_SOn(matrix))

#         matrix = np.random.rand(3,3)
#         # print(matrix)
#         # print(check_SOn(matrix))
#         matrix = correct_SOn(matrix)
#         # print(matrix)
#         # print(check_SOn(matrix))
#     elif method == 2:
#         vector = np.random.rand(4)
#         # print(vector)
#         # print(check_quaternion(vector))
#         vector = correct_quaternion(vector)
#         # print(vector)
#         # print(check_quaternion(vector))
#     elif method == 3:
#         matrix = np.random.rand(3,3)
#         # print(matrix)
#         # print("SE(2):", check_SEn(matrix))
#         matrix = correct_SEn(ma