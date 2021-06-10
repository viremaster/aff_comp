import math


def delta(point1, point2):
    return math.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))


with open("tsv.txt", "w") as file:
    file.write("Hello")
