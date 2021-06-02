#!/usr/bin/python3

import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, PointStamped
from tf2_geometry_msgs import do_transform_point
from PIL import Image
from numpy import asarray
from math import atan2

# check if pixel is valid
def is_valid(l):
    return l > 240


def check_for_new_square(data, size, x, y):
    # initial shape check
    if (data.shape[0] <= x + size) or (data.shape[1] <= y + size):
        return False

    # check if all pixles are valid
    for row in data[y:y+size]:
        for value in row[x:x+size]:
            if not is_valid(value):
                return False
    
    return True

def check_square_around_point(data, half_size, x, y):
    # assume that point is not on the edge of the image
    # check if all pixles are valid
    for row in data[y-half_size:y+half_size]:
        for value in row[x-half_size:x+half_size]:
            if not is_valid(value):
                return False
    
    return True


def square_exists(squares, x, y):
    for i, s in enumerate(squares):
        if (s[0] <= x and s[2] >= x) and (s[1] <= y and s[3] >= y):
            return i

    return -1


def sample_squares(data, size):
    squares = []
    x = 0
    y = 0

    while y < data.shape[1]:
        x = 0
        while x < data.shape[0]:
            exists = square_exists(squares, x, y)
            if exists >= 0:
                x = squares[exists][2] + 1
            else:
                if check_for_new_square(data, size, x, y):
                    squares.append((x,y,x+size,y+size))
                    x += size
                else:
                    x += 1

        y += 1

    return squares


def transform_map_to_world(shape, transform, x,y):
    res = 0.05

    x1 = x * res
    y1 = (shape[1] - y) * res

    pt = PointStamped()
    pt.point.x = x1
    pt.point.y = y1
    pt.point.z = 0

    transformed_pt = do_transform_point(pt, transform)
    return transformed_pt.point


def transform_map_to_world(shape, transform, x,y):
    res = 0.05

    x1 = x * res
    y1 = (shape[1] - y) * res

    pt = PointStamped()
    pt.point.x = x1
    pt.point.y = y1
    pt.point.z = 0

    transformed_pt = do_transform_point(pt, transform)
    return transformed_pt.point

def transform_world_to_map(x,y, shape):
    res = 0.05

    t = TransformStamped()
    t.transform.translation.x = 12.2
    t.transform.translation.y = 12.2
    t.transform.translation.z = 0.0
    t.transform.rotation = Quaternion(0,0,0,1)

    pt = PointStamped()
    pt.point.x = x
    pt.point.y = y
    pt.point.z = 0

    transformed_pt = do_transform_point(pt, t)
    x1 = transformed_pt.point.x / res
    y1 = shape[1] - (transformed_pt.point.y / res)

    #print("From: ", x, y)
    #print("To: ", x1, y1)
    return x1, y1


def is_point_valid(x, y, image_data):
    distance_from_edge = 8
    x1, y1 = transform_world_to_map(x, y, image_data.shape)
    if (image_data.shape[0] > x1 and image_data.shape[1] > y1):

        if (check_square_around_point(image_data, distance_from_edge, int(x1), int(y1))):
            # check square around point
            return True
    
    return False

def get_map_points(map_location):
    image = Image.open(map_location)
    image_data = asarray(image)
    square_size = 17

    squares = sample_squares(image_data, square_size)
    square_centers = [ ((s[0] + s[2])/2, (s[1]+s[3])/2) for s in squares]
    print(f"squares amount: {len(squares)}")

    map_transform = TransformStamped()
    map_transform.transform.translation.x = -12.2
    map_transform.transform.translation.y = -12.2
    map_transform.transform.translation.z = 0.0
    map_transform.transform.rotation = Quaternion(0,0,0,1)

    points = [transform_map_to_world(image_data.shape, map_transform, sc[0], sc[1]) for sc in square_centers]

    print(transform_world_to_map(0.883, -0.739, image_data.shape)) # 261., 250.
    return sorted([(p.x, p.y) for p in points], key=lambda p:atan2(p[0], p[1])), image_data







