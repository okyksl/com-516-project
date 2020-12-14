import math
import numpy as np


def slope(x, y):
    """Returns slope of the line from p1 to p2"""
    if x[0] == y[0]:
        return float('inf')
    else:
        return (x[1] - y[1])/ (x[0] - y[0])


def area(x, y, z):
    """Calculates area of the triangle formed by given triplet of points"""
    a = np.linalg.norm(x - y, ord=2)
    b = np.linalg.norm(y - z, ord=2)
    c = np.linalg.norm(z - x, ord=2)
    s = 0.5 * (a + b + c)
    return np.sqrt(s * (s-a) * (s-b) * (s-c))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Calculates convex hull of the given points
    
        Args:
            points np.ndarray (of shape n x 2)

        Returns:
            np.ndarray (of shape k x 2) containing the points in the convex hull
    """
    if len(points) < 4:
        return points

    # get points as list
    points = points.tolist()
    n = len(points)

    # sort point by first on x coord and then y coord
    points.sort(key=lambda x: [x[0], x[1]])

    # select point with min x as the start point
    start = points.pop(0)

    # sort points counterclockwise
    points.sort(key=lambda x: (slope(x, start), -x[1], x[0]))

    # get np representation
    points = np.array(points)

    # initialize hull
    hull = [ start, points[0] ]

    # consider points one by one
    for i in range(1, n-1):
        # remove if angle is concave
        while np.cross(hull[-1] - hull[-2], points[i] - hull[-2]) <= 0:
            hull.pop()
        hull.append(points[i])

    # remove last point if it is already inside
    if len(hull) > 3:
        if np.cross(hull[-1] - hull[-2], hull[0] - hull[-2]) <= 0:
            hull.pop()

    return np.array(hull)


def rotating_calipers(hull: np.ndarray) -> float:
    """Generates antipodal points of a given convex hull"""
    # handle trivial cases
    n = len(hull)
    if n == 0:
        return
    elif n == 1:
        yield hull[0], hull[0]
        return
    elif n == 2:
        yield hull[0], hull[1]
        return

    # find the first antipodal point
    j = 1
    while area(hull[n-1], hull[0], hull[j]) < area(hull[n-1], hull[0], hull[j+1]):
        j += 1

    # rotate while reporting antipodal points 
    end = j
    i = 0
    while i <= end and j < n:
        yield hull[i], hull[j]
        while j < n and area(hull[i], hull[(i+1) % n], hull[j]) < area(hull[i], hull[(i+1) % n], hull[(j+1) % n]):
            j += 1
            yield hull[i], hull[j % n]
        i += 1

def find_max_dist(points: np.ndarray) -> float:
    """Calculates the maximum distance between any pair of points"""
    # calculate convex hull
    hull = convex_hull(points)
    best = 0.

    # search through antipodal points
    for (x, y) in rotating_calipers(hull):
        best = np.maximum(best, np.linalg.norm(x - y, ord=2))
    return best

def find_max_dist_brute(points: np.ndarray) -> float:
    """Calculates the maximum distance between any pair of points"""
    n = points.shape[0]
    best = 0.
    for i in range(n):
        for j in range(i+1, n):
            best = np.maximum(best, np.linalg.norm(points[i] - points[j], ord=2))
    return best

