import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def fit_plane(*points):
    X = np.zeros([0, 3])
    Y = np.zeros([0, 1])
    for point in points:
        if point is not None:
            x, y, z = point
            X = np.concatenate([X, np.array([[1, x, z]])], axis=0)
            Y = np.concatenate([Y, np.array([[y]])], axis=0)
    Theta = np.linalg.inv((X.transpose().dot(X))).dot(X.transpose()).dot(Y)
    Theta = np.squeeze(Theta)
    A = Theta[1] / Theta[0]
    B = -1 / Theta[0]
    C = Theta[2] / Theta[0]
    return A, B, C


def get_transformed_points_on_plane(A, B, C, *points):
    P1, P2, P3, P4 = np.array(points[0]), np.array(points[1]), np.array(points[2]), np.array(points[3])

    z = np.array([A, B, C]) / (A**2 + B**2 + C**2)**0.5
    x = P4 - P1
    x = x / (np.sum(x**2))**0.5
    y = np.cross(z, x)
    R = np.concatenate([
        np.expand_dims(x, axis=1),
        np.expand_dims(y, axis=1),
        np.expand_dims(z, axis=1),
    ], axis=1)

    P1_ = R.dot(P1-P1)
    #P1_[-1] = 0
    if P2 is not None:
        P2_ = R.dot(P2-P1)
        #P2_[-1] = 0
    else:
        P2_ = None
    if P3 is not None:
        P3_ = R.dot(P3-P1)
        #P3_[-1] = 0
    else:
        P3_ = None
    P4_ = R.dot(P4-P1)
    #P4_[-1] = 0
    return P1_, P2_, P3_, P4_


def get_transformed_points_on_plane_2(A, B, C, *points):
    pointsp = []
    for point in points:
        x, y, z = point
        xp = ((B**2 + C**2)*x - A*(B*y + C*z + 1)) / (A**2 + B**2 + C**2)
        yp = ((A**2 + C**2)*y - B*(A*x + C*z + 1)) / (A**2 + B**2 + C**2)
        zp = ((A**2 + B**2)*z - C*(A*x + B*y + 1)) / (A**2 + B**2 + C**2)
        pointsp.append((xp, yp, zp))
    P1p, P2p, P3p, P4p = map(np.array, pointsp)  # √
    X = P4p - P1p
    lenX = np.sum(X**2)**0.5
    V2 = P2p - P1p
    x2 = V2.dot(X) / lenX
    y2 = (np.sum(V2**2) - x2**2)**0.5
    V3 = P3p - P1p
    x3 = V3.dot(X) / lenX
    y3 = (np.sum(V3**2) - x3**2)**0.5
    # 判断同侧还是异侧
    eps = 1e-5
    vec = P3p - P2p
    if abs((y2 + y3)**2 + (x2 - x3)**2 - np.sum(vec**2)) < eps:
        # 异侧
        y3 = -y3
    elif abs((y2 - y3)**2 + (x2 - x3)**2 - np.sum(vec**2)) < eps:
        # 同侧
        pass
    else:
        return [False, None, None, None, None]
    coord = [(0, 0), (x2, y2), (x3, y3), (lenX, 0)]
    return [True, *coord]


def fit_parabola_on_plane(*points):
    X = np.zeros([0, 3])
    Y = np.zeros([0, 1])
    for point in points:
        if point is not None:
            x, y = point[0], point[1]
            X = np.concatenate([X, np.array([[1, x, x**2]])], axis=0)
            Y = np.concatenate([Y, np.array([[y]])], axis=0)
    Theta = np.linalg.inv((X.transpose().dot(X))).dot(X.transpose()).dot(Y)
    Theta = np.squeeze(Theta)
    a, b, c = Theta[0], Theta[1], Theta[2]  # y = a + b*x +c*x^2
    return a, b, c


def cal_fitted_length(*points, log=False):
    if log:
        print("P1: {}, P2: {}, P3: {}, P4: {}".format(points[0], points[1], points[2], points[3]))
    A, B, C = fit_plane(*points)
    if log:
        print("A: {}, B: {}, C: {}".format(A, B, C))
    # P1, P2, P3, P4 = get_transformed_points_on_plane(A, B, C, *points)
    can_project, P1, P2, P3, P4 = get_transformed_points_on_plane_2(A, B, C, *points)
    if not can_project:
        return None
    if log:
        print("P1_: {}, P2_: {}, P3_: {}, P4_: {}".format(P1, P2, P3, P4))

    # projected_length = sum((np.array(P1) - np.array(P2))**2)**0.5 + sum((np.array(P2) - np.array(P3))**2)**0.5 + sum((np.array(P3) - np.array(P4))**2)**0.5
    # print("projected length:", projected_length)
    # This is right!

    x0, x1, x2, x3 = P1[0], P2[0], P3[0], P4[0]
    a, b, c = fit_parabola_on_plane(P1, P2, P3, P4)  # √
    if log:
        print("a: {}, b: {}, c: {}".format(a, b, c))
    # ans, error = quad(lambda t: ((xT * t)**2 + (a + b*xT * t + (c*xT**2) * t**2)**2)**0.5, 0, 1)
    ans, error = quad(lambda x: (1 + b**2 + 4*b*c*x + 4*(c**2)*x**2) ** 0.5, 0, x3)
    return ans


def cal_poly_length(*points):
    points = [point for point in points if point is not None]
    l = 0
    for i in range(len(points)-1):
        l += sum((np.array(points[i]) - np.array(points[i+1]))**2)**0.5
    return l


if __name__ == "__main__":
    points = (
        (1, 2, 3),
        (3, 4, 5),
        (9, 4, 7),
        (8, 4, 2),
    )
    print(cal_fitted_length(*points))
    pass
