import os
import math
import sys
import vtk
import pathlib
import numpy as np
from timeit import default_timer as timer
from multiprocessing import Pool

from collections import defaultdict
from vtk.util import numpy_support as npsup

'''

'''
def get_plane_size(image_size):
    #Make a meshgrid with the limits of the image
    uu, vv = np.meshgrid([-image_size, image_size], [-image_size, image_size], sparse=True)
    return  uu, vv

def get_ray_length(image_size):
    return np.sqrt(3) * image_size

def points_on_sphere(dim, N, norm=np.random.normal):
    """
    http://en.wikipedia.org/wiki/N-sphere#Generating_random_points
    """
    #For whatever reason, when you pass 1 it always returns a -1, 1, 1, as the points
    if N == 1:
        N += 1
        normal_deviates = norm(size=(N, dim))
        normal_deviates = normal_deviates[1]
    else:
        normal_deviates = norm(size=(N, dim))
    radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
    points = normal_deviates / radius
    return points, normal_deviates

def get_plane_from_points(points, uu, vv):
    u = (0, 1, 0) if np.allclose(points, (1, 0, 0)) else np.cross(points, (1, 0, 0))
    u /= math.sqrt((u ** 2).sum())
    v = np.cross(points, u)
    u = u[:, np.newaxis, np.newaxis]
    v = v[:, np.newaxis, np.newaxis]
    xx, yy, zz = u * uu + v * vv
    print(xx, yy, zz)
    return xx, yy, zz

def get_point_array(xx, yy, zz, normal, spacing):
    x = np.arange(xx.min(), xx.max(), int(spacing))
    y = np.arange(yy.min(), yy.max(), int(spacing))
    xv, yv = np.meshgrid(x, y)
    v = normal
    z = (-v[0] * xv.flatten() - v[1] * yv.flatten())/v[2]
    x_points = xv.flatten()
    y_points = yv.flatten()
    z_points = z
    point_array = np.column_stack([x_points, y_points, z_points])
    point_normals = v
    return point_array, point_normals

def isHit(obbTree, pSource, pTarget):
    """
    From https://blog.kitware.com/ray-casting-ray-tracing-with-vtk/
    :param obbTree:
    :param pSource:
    :param pTarget:
    :return:
    """
    code = obbTree.IntersectWithLine(pSource, pTarget, None, None)
    if code == 0:
        return False
    return True

def GetIntersect(obbTree, pSource, pTarget):
    '''
    From https://blog.kitware.com/ray-casting-ray-tracing-with-vtk/
    :param obbTree:
    :param pSource:
    :param pTarget:
    :return:
    '''
    points = vtk.vtkPoints()
    cellIds = vtk.vtkIdList()

    #Perform intersection test
    code = obbTree.IntersectWithLine(pSource, pTarget, points, cellIds)
    if code == 1:
        pointData = points.GetData()
        noPoints = pointData.GetNumberOfTuples()
        noIds = cellIds.GetNumberOfIds()
        pointsInter = {pointData.GetTuple3(idx) for idx in range(noPoints)}
        cellIdsInter = {cellIds.GetId(idx) for idx in range(noPoints)}

        return pointsInter, cellIdsInter


def vtk_merge(pd1, pd2):
    """
    merge two PolyDataSets into one.
    @param vtkPolyData pd1  the first dataset
    @param vtkPolyData pd2  the second dataset

    @return vtkPolyData the merged dataset
    """
    a = vtk.vtkAppendPolyData()
    a.AddInput(pd1)
    a.AddInput(pd2)
    return a.GetOutput()

def loadSTL(filenameSTL):
    readerSTL = vtk.vtkSTLReader()
    readerSTL.SetFileName(filenameSTL)
    # 'update' the reader i.e. read the .stl file
    readerSTL.Update()

    polydata = readerSTL.GetOutput()

    # If there are no points in 'vtkPolyData' something went wrong
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(
            "No point data could be loaded from '" + filenameSTL)
        return None

    return polydata

def vtk_intersections(mesh, point_array, point_normals, raycast_length=""):
    point_normals = point_normals
    reverse_normals = -1 * point_normals

    # Load the points into a vtk object
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(npsup.numpy_to_vtk(point_array , deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtkPts)
    noPoints = poly.GetNumberOfPoints()

    #Set up the length of all the lines
    if raycast_length == "":
        image_size = abs(np.array(mesh.GetBounds()).min() - np.array(mesh.GetBounds()).max())
        RayCastLength = get_ray_length(image_size)
    else:
        RayCastLength = 128 * 128

    #initiate the mesh with the vtk OBB tree
    mesh_voi = vtk.vtkOBBTree()
    mesh_voi.SetDataSet(mesh)
    mesh_voi.BuildLocator()

    # Set up a list to store the results
    total_points_list = set()
    total_cells_list = set()

    pointRayStart = [poly.GetPoint(idx) + RayCastLength * np.array(reverse_normals) for idx in range(noPoints)]
    pointRayTarget = [poly.GetPoint(idx) + RayCastLength * np.array(point_normals) for idx in range(noPoints)]

    points_list = [GetIntersect(mesh_voi, pointRayStart[idx], pointRayTarget[idx]) for idx in range(noPoints)]
    #It's much faster just to drop the empty sets
    points_list = [x for x in points_list if x]

    for idx in range(len(points_list)):
        total_cells_list.update(points_list[idx][1])
        total_points_list.update(points_list[idx][0])
    return total_points_list, total_cells_list

def vtk_get_mesh_info(vtk_poly):
    if vtk_poly.GetNumberOfPolys() == 0:
        return 0
    Mass = vtk.vtkMassProperties()
    Mass.SetInputData(vtk_poly)
    Mass.Update()
    return Mass.GetVolume(), Mass.GetSurfaceArea()

def getMinVolEllipse(P, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points

    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!

    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]

    Returns:
    (center, radii, rotation)

    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT, np.dot(linalg.inv(V), Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
        np.dot(P.T, np.dot(np.diag(u), P)) -
        np.array([[a * b for b in center] for a in center])
    ) / d

    # Get the values we'd like to return
    U, eigen_vals, rotation = linalg.svd(A)
    radii = 1.0 / np.sqrt(eigen_vals)

    return (eigen_vals, U, center, radii, rotation)

def mvee(points, tol=0.0001, flag='outer'):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = 1
    u = np.ones(N) / N
    #  inner ellipse: if d < 1+tol_dist
    #  outer ellipse : while err > tol:
    if flag == 'inner':
        while err < 1 + tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
    elif flag == 'outer':
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
    c = u * points
    A = la.inv(points.T * np.diag(u) * points - c.T * c) / d
    return np.asarray(A), np.squeeze(np.asarray(c))

'''
Script starts

'''
filename = r'D:\Desktop\quanfima\data\results\mesh.stl'

mesh = loadSTL(filename)

#points = vtk.vtkPoints()
#cellIds = vtk.vtkIdList()
#line = (0.0, 0.0, 0.0), (127.0, 127.0, 127.0)

#code = obbTree.IntersectWithLine(line[0], line[1], points, cellIds)

#How many planes to generate
n = 50

#The dimensionality
d = 3

image_size = abs(np.array(mesh.GetBounds()).min() - np.array(mesh.GetBounds()).max())

#Set up the sampling plane size so it is the size of the inter diagnonal
uu, vv =  get_plane_size(image_size)

#Generate points for the plane using  randomly generated normals on a spehere
rand_points, normals = points_on_sphere(dim=d, N=n, norm=np.random.normal)

total_points_list = []
point_used = []
#Loop acroos the random points and generate plane
for number, p in enumerate(rand_points):
    start = timer()
    xx, yy, zz = get_plane_from_points(p, uu, vv)
    #Get points along the plne
    point_array, point_normals = get_point_array(xx, yy, zz, normal=normals[number], spacing=10)
    point_used.append(len(point_array))
    print(len(point_array))
    points_intersection = vtk_intersections(mesh=mesh,
                            point_array=point_array,
                            point_normals=point_normals,
                            raycast_length="")
    total_points_list.append(points_intersection)
    end = timer()
    print("Finding intersections took", abs(start - end))

MIL = [np.sum(point_used[idx] * (2 * get_ray_length(image_size)))/(len(total_points_list[idx][0]) * 0.5) for idx in range(len(rand_points))]
point_cloud = [MIL[idx] * normals[idx] for idx in range(len(rand_points))]
df = pd.DataFrame(point_cloud)
df = df[np.isfinite(df).all(1)]

new_points = set()
new_cells = set()
for idx in range(len(total_points_list)):
    new_points.update(total_points_list[idx][0])
    new_cells.update(total_points_list[idx][1])


final_points = pd.DataFrame.from_records(pd.DataFrame(points_list).T.unstack().dropna().values)


def unpack_intersections(args):
    """Unpack input arguments and return result of `execute_tensor` function
    """
    return vtk_intersections(*args)


args = zip(mesh=mesh, point_array=point_array, point_normals=point_normals, reverse_normals=reverse_normals, raycast_length="")

proc_pool = Pool(processes=7)
results = np.array(proc_pool.map(unpack_intersections, args))
#results = np.array(proc_pool.map(unpack_intersections, args))
proc_pool.close()
proc_pool.join()
proc_pool.terminate()

#https://stackoverflow.com/questions/53747259/ray-tracing-and-cuda

'''
my_mesh = mesh.Mesh.from_file(r'D:\Desktop\quanfima\data\results\mesh.stl')
n = my_mesh.normals
v0 = my_mesh.v0
v1 = my_mesh.v1
v2 = my_mesh.v2

mod = SourceModule("""
    #include <math.h>
    //#include <vector>
  __global__ void intersect(float *origin,
                            float *dir,
                            float *v0,
                            float *v1,
                            float *v2,
                            float *int_point_real)
  {
    using namespace std;
    //#include <vector>
    //#include <math.h>
    int idx = threadIdx.x;
    //a[idx] *= 2;
    int count = 0;

    //std::vector<double> v0_current(3);
    float v0_current[3];
    float v1_current[3];
    float v2_current[3];
    float dir_current[3] = {dir[0],dir[1],dir[2]}; 
    //std::vector<double> v1_current(3);
    //std::vector<double> v2_current(3);
    float int_point[3];
    //std::vector<float> int_point(3);
    //std::vector<std::vector<float>> int_pointS;
    float int_pointS[2][3];
    //std::vector<std::vector<double>> int_point;
    //std::vector<int> int_faces;
    int int_faces[2];
    float dist[2];
    //std::vector<float> dist;
    int n_tri = 960;

    for(int i = 0; i<n_tri; i++) {
        for (int j = 0; j<3; j++){
            v0_current[j] = v0[j];
            v1_current[j] = v1[j];
            v2_current[j] = v2[j];
        }
        double eps = 0.0000001;
        //std::vector<float> E1(3);
        float E1[3];
        //std::vector<float> E2(3);
        float E2[3];
        //std::vector<float> s(3);
        float s[3];
        for (int j = 0; j < 3; j++) {
            E1[j] = v1_current[j] - v0_current[j];
            E2[j] = v2_current[j] - v0_current[j];
            s[j] = origin[j] - v0_current[j];
        }
        //std::vector<float> h(3);
        float h[3];
        h[0] = dir[1] * E2[2] - dir[2] * E2[1];
        h[1] = -(dir[0] * E2[2] - dir[2] * E2[0]);
        h[2] = dir[0] * E2[1] - dir[1] * E2[0];
        float a;
        a = E1[0] * h[0] + E1[1] * h[1] + E1[2] * h[2];
        if (a > -eps && a < eps) {
            int_point[0] = false;
            //return false;
        }
        else {
            double f = 1 / a;
            float u;
            u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
            if (u < 0 || u > 1) {
                int_point[0] = false;
                //return false;
            }
            else {
                //std::vector<float> q(3);
                float q[3];
                q[0] = s[1] * E1[2] - s[2] * E1[1];
                q[1] = -(s[0] * E1[2] - s[2] * E1[0]);
                q[2] = s[0] * E1[1] - s[1] * E1[0];
                float v;
                v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
                if (v < 0 || (u + v)>1) {
                    int_point[0] = false;
                    //return false;
                }
                else {
                    float t;
                    t = f * (E2[0] * q[0] + E2[1] * q[1] + E2[2] * q[2]);
                    if (t > eps) {
                        for (int j = 0; j < 3; j++) {
                            int_point[j] = origin[j] + dir_current[j] * t;
                        }
                        //return t;
                    }
                }
            }
        }
        if (int_point[0] != false) {
            count = count+1;
            //int_faces.push_back(i);
            int_faces[count-1] = i;
            //dist.push_back(sqrt(pow((origin[0] - int_point[0]), 2) + pow((origin[1] - int_point[1]), 2) + pow((origin[2] - int_point[2]), 2)));
            //dist.push_back(x);
            dist[count-1] = sqrt(pow((origin[0] - int_point[0]), 2) + pow((origin[1] - int_point[1]), 2) + pow((origin[2] - int_point[2]), 2));
            //int_pointS.push_back(int_point);
            for (int j = 0; j<3; j++) {
                int_pointS[count-1][j] = int_point[j];
            }
        } 
    }
    double min = dist[0];
    int ind_min = 0;
    for (int i = 0; i < int_pointS.size(); i++){
        if (min > dist[i]) {
            min = dist[i];
            ind_min = i;
        }
    }
    //dist_real[Idx] = dist[ind_min];
    //int_point_real_x[Idx] = int_pointS[ind_min][0];
    //int_point_real_y[Idx] = int_pointS[ind_min][1];
    //int_point_real_z[Idx] = int_pointS[ind_min][2];
    int_point_real[0] = int_pointS[ind_min][0];
    int_point_real[1] = int_pointS[ind_min][1];
    int_point_real[2] = int_pointS[ind_min][2];
}
  """)

origin = np.asarray([1, 1, 1]).astype(np.float32)
direction = np.ones((100, 3)).astype(np.float32)
int_point_real = np.zeros((100, 3)).astype(np.float32)

intersect = mod.get_function("intersect")
intersect(drv.In(origin), drv.In(direction), drv.In(v0), drv.In(v1), drv.In(v2), drv.Out(int_point_real), block=(512,1,1), grid=(64,1,1))

#import pycuda.driver as drv
#import pycuda.autoinit
#from pycuda.compiler import SourceModule

# !/usr/bin/env python
# -*- coding: utf-8 -*-

SPHERE SCENE RAY TRACER
This is a basic port of the simple ray tracer from Chapter 6 of
"CUDA by Example", by Sanders and Kandrot. With a few exceptions
(notably, the `hit()` method is not bound to a struct containing 
sphere data and we use a numpy record array).
On my GeForce GTX 750ti, the kernel computes in ~180 microseconds,
and the entire operation including data transfers takes about 
100 milliseconds. That's ~0.2x the speed of the pure CUDA 
implementation.
Author: Daniel Rothenberg <darothen@mit.edu>
https://gist.github.com/darothen/f53bb3e40edbceb38904


import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from numba import cuda, int16, float32, from_dtype
from timeit import default_timer as timer

DIM = 600  # domain width, in pixels
RAYS = 200  # number of rays to cast
INF = 2e10  # really large number
VERBOSE = True  # print out some debug stuff along the way

# A numpy record array (like a struct) to record sphere data
Square = np.dtype([
    # square (x, y, z) coordinates
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ], align=True)


Square_t = from_dtype(Square)  # Create a type that numba can recognize!

@jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

# We can use that type in our device functions and later the kernel!
@cuda.jit(restype=float32, argtypes=[float32, float32, Square_t],
          device=True, inline=False)
def hit(ox, oy, pix):
    """ Compute whether a ray parallel to the z-axis originating at
    (ox, oy, INF) will intersect a given sphere; if so, return the
    distance to the surface of the sphere.
    """
    dx = ox - pix.x
    dy = oy - pix.y
    if dx <= 0 or dy <= 0:
        return -INF
    else:
        dz = (dx, dx)
        return dz

#Sphere_t is the data type
@cuda.jit(argtypes=(Square_t[:], int16[:, :, :]))
def kernel(pix, bitmap):
    x, y = cuda.grid(2)
    # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
    #           threadIdx.y + ( blockIdx.y * blockDim.y )
    # shift the grid to [-DIM/2, DIM/2]
    ox = x - DIM / 2
    oy = y - DIM / 2
    maxz = -INF

    i = 0  # emulate a C-style for-loop, exposing the idx increment logic
    while (i < pix):
        t = hit(ox, oy, spheres[i])
        rad = spheres[i].radius

        if (t > maxz):
            dz = t - spheres[i].z  # t = dz + z; inverting hit() result
            n = dz / sqrt(rad * rad)
            fscale = n  # shades the color to be darker as we recede from
            # the edge of the cube circumscribing the sphere

            r = spheres[i].r * fscale
            g = spheres[i].g * fscale
            b = spheres[i].b * fscale
            maxz = t
        i += 1

    # Save the RGBA value for this particular pixel
    bitmap[x, y, 0] = int(r * 255.)
    bitmap[x, y, 1] = int(g * 255.)
    bitmap[x, y, 2] = int(b * 255.)
    bitmap[x, y, 3] = 255

"""
Original example
"""

DIM = 2048  # domain width, in pixels
DM = min([DIM, 1000])  # constraint for sphere locations
SPHERES = 200  # number of spheres in scene
INF = 2e10  # really large number
VERBOSE = True  # print out some debug stuff along the way

# Randomly generate a number between [0, x)
rnd = lambda x: x * np.random.rand()

# A numpy record array (like a struct) to record sphere data
Sphere = np.dtype([
    # RGB color values (floats from [0, 1])
    ('r', 'f4'), ('g', 'f4'), ('b', 'f4'),
    # sphere radius
    ('radius', 'f4'),
    # sphere (x, y, z) coordinates
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ], align=True)

Sphere_t = from_dtype(Sphere)  # Create a type that numba can recognize!

@cuda.jit(restype=float32, argtypes=[float32, float32, Sphere_t],
          device=True, inline=False)
def hit(ox, oy, sph):
    """ Compute whether a ray parallel to the z-axis originating at
    (ox, oy, INF) will intersect a given sphere; if so, return the
    distance to the surface of the sphere.
    """
    dx = ox - sph.x
    dy = oy - sph.y
    rad = sph.radius
    if (dx * dx + dy * dy < rad * rad):
        dz = sqrt(rad * rad - dx * dx - dy * dy)
        return dz + sph.z
    else:
        return -INF

#Sphere_t is the data type
@cuda.jit(argtypes=(Sphere_t[:], int16[:, :, :]))
def kernel(spheres, bitmap):
    x, y = cuda.grid(2)  # alias for threadIdx.x + ( blockIdx.x * blockDim.x ),
    #           threadIdx.y + ( blockIdx.y * blockDim.y )
    # shift the grid to [-DIM/2, DIM/2]
    ox = x - DIM / 2
    oy = y - DIM / 2

    r = 0.
    g = 0.
    b = 0.
    maxz = -INF

    i = 0  # emulate a C-style for-loop, exposing the idx increment logic
    while (i < SPHERES):
        t = hit(ox, oy, spheres[i])
        rad = spheres[i].radius

        if (t > maxz):
            dz = t - spheres[i].z  # t = dz + z; inverting hit() result
            n = dz / sqrt(rad * rad)
            fscale = n  # shades the color to be darker as we recede from
            # the edge of the cube circumscribing the sphere

            r = spheres[i].r * fscale
            g = spheres[i].g * fscale
            b = spheres[i].b * fscale
            maxz = t
        i += 1

    # Save the RGBA value for this particular pixel
    bitmap[x, y, 0] = int(r * 255.)
    bitmap[x, y, 1] = int(g * 255.)
    bitmap[x, y, 2] = int(b * 255.)
    bitmap[x, y, 3] = 255



start = timer()

# Create a container for the pixel RGBA information of our image
bitmap = np.zeros([DIM, DIM, 4], dtype=np.int16)

# Copy to device memory
d_bitmap = cuda.to_device(bitmap)

# Create empty container for our Sphere data on device
d_spheres = cuda.device_array(SPHERES, dtype=Sphere_t)

# Create an empty container of spheres on host, and populate it
# with some random data.
temp_spheres = np.empty(SPHERES, dtype=Sphere_t)
for i in range(SPHERES):
    temp_spheres[i]['r'] = rnd(1.0)
    temp_spheres[i]['g'] = rnd(1.0)
    temp_spheres[i]['b'] = rnd(1.0)
    temp_spheres[i]['x'] = rnd(DIM) - DIM / 2
    temp_spheres[i]['y'] = rnd(DIM) - DIM / 2
    temp_spheres[i]['z'] = rnd(DIM) - DIM / 2
    temp_spheres[i]['radius'] = rnd(100.0) + 20

    if VERBOSE:
        sph = temp_spheres[i]
        print("Sphere %d" % i)
        print("\t(r,g,b)->(%1.2f,%1.2f,%1.2f)" % (sph['r'], sph['b'], sph['g']))
        print("\t(x,y,z)->(%4.1f,%4.1f,%4.1f)" % (sph['x'], sph['y'], sph['z']))
        print("\tradius->%3.1f" % sph['radius'])

# Copy the sphere data to the device
cuda.to_device(temp_spheres, to=d_spheres)

# Here, we choose the granularity of the threading on our device. We want
# to try to cover the entire image with simulatenous threads, so we'll
# choose a grid of (DIM/16. DIM/16) blocks, each with (16, 16) threads
grids = (int(DIM / 16), int(DIM / 16))
threads = (16, 16)

# Execute the kernel
kernel[grids, threads](d_spheres, d_bitmap)
kernel_dt = timer() - start

# Copy the result from the kernel ordering the ray tracing back to host
bitmap = d_bitmap.copy_to_host()
mem_dt = timer() - start

print("Elapsed time in")
print("          kernel:  {:3.1f} µs".format(kernel_dt * 1e6))
print("    device->host:  {:3.1f} ms".format(mem_dt * 1e3))

# Visualize the resulting scene. We'll do this with two side-by-side plots:
#    Left -> the scene rendered in psuedo-3D, accounting for sphere placement
#   Right -> flat circle projections of spheres, in z-order looking down

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

bitmap = np.transpose(bitmap / 255., (1, 0, 2))  # swap image's x-y axes
axs[0].imshow(bitmap)
axs[0].grid(False)

# sort the spheres by Z for visualizing z-level order, plot using circle artists
temp_spheres.sort(order='radius')
for i in range(SPHERES):
    sph = temp_spheres[-i]  # temp_spheres is actually backwards!
    circ = plt.Circle((sph['x'] + DIM / 2, sph['y'] + DIM / 2), sph['radius'],
                      color=(sph['r'], sph['g'], sph['b']))
    axs[1].add_artist(circ)

for ax in axs:
    ax.set_xlim(0, DIM)
    ax.set_ylim(0, DIM)

plt.show()
'''