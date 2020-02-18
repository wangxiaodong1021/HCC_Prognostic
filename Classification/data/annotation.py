# -*-coding:utf-8-*-
import json
import copy
import json
import numpy as np
from skimage.measure import points_in_poly


np.random.seed(0)


class Polygon(object):
    """
    Polygon represented as [N, 2] array of vertices
    """
    def __init__(self, name, vertices):
        """
        Initialize the polygon.

        Arguments:
            name: string, id of the polygon
            vertices: [N, 2] 2D numpy array of int
        """
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        """
        Determine if a given coordinate is inside the polygon or not.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the polygon.
        """
        # print("type:",[coord],self._vertices.shape)
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)

    def length(self):
        return self._vertices.shape[0]


class Annotation(object):
    """
    Annotation about the regions within WSI in terms of vertices of polygons.
    """
    def __init__(self):
        self._json_path = ''
        self._polygons_tumor = []
        self._polygons_tumor_beside = []
        self._polygons_fibrous_tissue = []
        self._polygons_necrosis = []
        self._polygons_lymph = []
        self._color_lymph = '#bd10e0'
        self._color_tumor = '#f5a623'  # 癌症 标注颜色为  '#f5a623' 橙色
        self._color_tumor_beside = ['#b8e986', '#7ed321']  # 癌旁 绿色
        self._color_fibrous_tissue = '#f8e71c'  # 纤维组织 黄色
        self._color_necrosis = '#d0021b'  # 坏死  红色

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        """
        Initialize the annotation from a json file.

        Arguments:
            json_path: string, path to the json annotation.
        """
        self._json_path = json_path

        with open(json_path, 'r') as f:
            annotations_json = json.load(f)
        cnt = -1
        for mark in annotations_json:
            cnt += 1
            if mark["color"] == self._color_lymph:
                coord = mark["coordinates"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    for c in value:
                        vertices.append([c[0], c[1]])
                vertices = np.array(vertices)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['id']), vertices)
                    self._polygons_lymph.append(polygon)
            elif mark["color"] == self._color_tumor:
                coord = mark["coordinates"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    for c in value:
                        vertices.append([c[0], c[1]])
                vertices = np.array(vertices)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['id']), vertices)
                    self._polygons_tumor.append(polygon)
            elif mark["color"] in self._color_tumor_beside:
                coord = mark["coordinates"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    for c in value:
                        vertices.append([c[0], c[1]])
                vertices = np.array(vertices)
                if len(vertices) == 0:
                    print("vertices is none:", vertices, self._json_path)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['id']), vertices)
                    self._polygons_tumor_beside.append(polygon)
            elif mark["color"] == self._color_fibrous_tissue:
                coord = mark["coordinates"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    for c in value:
                        vertices.append([c[0], c[1]])
                vertices = np.array(vertices)
                if len(vertices) == 0:
                    print("vertices is none:", vertices, self._json_path)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['id']), vertices)
                    self._polygons_fibrous_tissue.append(polygon)
            elif mark["color"] == self._color_necrosis:
                coord = mark["coordinates"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    for c in value:
                        vertices.append([c[0], c[1]])
                vertices = np.array(vertices)
                if len(vertices) == 0:
                    print("vertices is none:", vertices, self._json_path)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['id']), vertices)
                    self._polygons_necrosis.append(polygon)
            else:
                print("Error:Abnormal classification!", self._json_path, mark['color'])

    def inside_polygons(self, kind, coord):
        """
        Determine if a given coordinate is inside the tumor/tumor_bedide/
        fibrous_tissue/necrosis polygons of the annotation.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        if kind == 'lymph':
            polygons = copy.deepcopy(self._polygons_lymph)
        elif kind == 'tumor':
            polygons = copy.deepcopy(self._polygons_tumor)
        elif kind == 'tumor_beside':
            polygons = copy.deepcopy(self._polygons_tumor_beside)
        elif kind == 'fibrous_tissue':
            polygons = copy.deepcopy(self._polygons_fibrous_tissue)
        elif kind == 'necrosis':
            polygons = copy.deepcopy(self._polygons_necrosis)
        polygon_cp = []
        for polygon in polygons:
            if polygon.inside(coord):
                polygon_cp = polygon
                return True, polygon_cp
        return False, polygon_cp

    def polygon_vertices(self):
        """
        Return the polygon represented as [N, 2] array of vertices

        Arguments:
            is_positive: bool, return positive or negative polygons.

        Returns:
            [N, 2] 2D array of int
        """
        return list(map(lambda x: x.vertices(), self._polygons))

# class Formatter(object):
#     """
#     Format converter e.g. CAMELYON16 to internal json
#     """
#     def camelyon16xml2json(inxml, outjson):
#         """
#         Convert an annotation of camelyon16 xml format into a json format.
#
#         Arguments:
#             inxml: string, path to the input camelyon16 xml format
#             outjson: string, path to the output json format
#         """
#         root = ET.parse(inxml).getroot()
#         annotations_tumor = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
#         annotations_0 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
#         annotations_1 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
#         annotations_2 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
#         annotations_positive = \
#             annotations_tumor + annotations_0 + annotations_1
#         annotations_negative = annotations_2
#
#         json_dict = {}
#         json_dict['positive'] = []
#         json_dict['negative'] = []
#
#         for annotation in annotations_positive:
#             X = list(map(lambda x: float(x.get('X')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             Y = list(map(lambda x: float(x.get('Y')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             vertices = np.round([X, Y]).astype(int).transpose().tolist()
#             name = annotation.attrib['Name']
#             json_dict['positive'].append({'name': name, 'vertices': vertices})
#
#         for annotation in annotations_negative:
#             X = list(map(lambda x: float(x.get('X')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             Y = list(map(lambda x: float(x.get('Y')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             vertices = np.round([X, Y]).astype(int).transpose().tolist()
#             name = annotation.attrib['Name']
#             json_dict['negative'].append({'name': name, 'vertices': vertices})
#
#         with open(outjson, 'w') as f:
#             json.dump(json_dict, f, indent=1)
#
#     def vertices2json(outjson, positive_vertices=[], negative_vertices=[]):
#         json_dict = {}
#         json_dict['positive'] = []
#         json_dict['negative'] = []
#
#         for i in range(len(positive_vertices)):
#             name = 'Annotation {}'.format(i)
#             vertices = positive_vertices[i].astype(int).tolist()
#             json_dict['positive'].append({'name': name, 'vertices': vertices})
#
#         for i in range(len(negative_vertices)):
#             name = 'Annotation {}'.format(i)
#             vertices = negative_vertices[i].astype(int).tolist()
#             json_dict['negative'].append({'name': name, 'vertices': vertices})
#
#         with open(outjson, 'w') as f:
#             json.dump(json_dict, f, indent=1)
