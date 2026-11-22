"""Native two-sided surface FEM. Seam and USD patterns adapted from PR #2126, MIT."""

from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np


def _refine_edges(vertices, faces, eligible):
    """Split contact-zone edges and their neighbours without introducing T junctions."""
    points = list(vertices)
    midpoint = {}

    def split(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoint:
            p, q = np.asarray(points[a]), np.asarray(points[b])
            midpoint[key] = None
            if eligible(p, q):
                midpoint[key] = len(points)
                points.append(tuple((p+q)/2))
        return midpoint[key]

    refined = []
    for a, b, c in faces:
        ab, bc, ca = split(a, b), split(b, c), split(c, a)
        count = sum(m is not None for m in (ab, bc, ca))
        if count == 0:
            refined.append((a, b, c))
        elif count == 3:
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        elif count == 1:
            while ab is None:
                a, b, c, ab, bc, ca = b, c, a, bc, ca, ab
            refined.extend(((a, ab, c), (ab, b, c)))
        else:
            while ca is not None:
                a, b, c, ab, bc, ca = b, c, a, bc, ca, ab
            refined.extend(((a, ab, c), (ab, bc, c), (ab, b, bc)))
    return points, refined


def shirt_mesh(refine_corner=False, *, refine_z_buckle=False):
    """Return adult cut pattern in metres, with two shell faces and sewn edges."""
    vertices, triangles, indices = [], [], {}

    def vertex(i, j):
        if (i, j) not in indices:
            indices[i, j] = len(vertices)
            vertices.append((i / 100, j / 100))
        return indices[i, j]

    for i in range(68):
        for j in range(-37, 37):
            x, y = i + 0.5, abs(j + 0.5)
            torso = y < 25
            sleeve = 49 + (y - 25) / 3 < x < 67 - (y - 25) / 4
            neck = y < 8 and x > 62 + (y / 8) ** 2 * 5
            if (torso or sleeve) and not neck:
                a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
                triangles.extend(((a, b, c), (a, c, d)))

    if refine_corner:
        for _ in range(2):
            vertices, triangles = _refine_edges(vertices, triangles, lambda p, q:
                max(p[0], q[0]) <= .04 and min(p[1], q[1]) >= .21)
    if refine_z_buckle:
        for _ in range(2):
            vertices, triangles = _refine_edges(vertices, triangles, lambda p, q:
                .26 <= min(p[0], q[0]) < max(p[0], q[0]) <= .32)
    xy = np.array(vertices)
    n = len(xy)
    directed_edges = [(a, b) for a, b, c in triangles for a, b in ((a, b), (b, c), (c, a))]
    edges = Counter(tuple(sorted(edge)) for edge in directed_edges)
    rest = np.vstack(tuple(np.c_[xy, np.full(n, z)] for z in (0.001, -0.001)))
    faces = list(triangles) + [(a+n, c+n, b+n) for a, b, c in triangles]
    for a, b in directed_edges:
        if edges[tuple(sorted((a, b)))] != 1:
            continue
        x, y = (xy[a] + xy[b]) / 2
        opening = x == 0 or abs(y) > 0.369 or (x >= 0.62 and abs(y) <= 0.0801)
        if not opening:
            faces.extend(((b, a, a+n), (b, a+n, b+n)))
    return rest, np.array(faces), len(triangles)


def wrinkled_start(rest, mat_top, fold_angle=0.):
    """Start with the lower hem turned back on itself, supported by self-contact."""
    # A lap, unlike a freestanding elastic ridge, can remain folded under gravity.
    # Parameterize by fabric length so the fold does not rely on stretching.
    c, d = np.cos(fold_angle), np.sin(fold_angle)
    s = c*rest[:, 0] + d*rest[:, 1]
    across = -d*rest[:, 0] + c*rest[:, 1]
    lap, bend = 0.11, 0.04
    radius = bend / np.pi
    theta = np.pi + np.pi * np.clip((s - lap) / bend, 0., 1.)
    curve_x = radius * np.sin(theta)
    curve_z = radius * (1. + np.cos(theta - np.pi))
    x = curve_x + np.maximum(lap - s, 0.) + np.maximum(s - lap - bend, 0.)
    points = rest.copy()
    along = lap + bend + x - rest[:, 2] * np.sin(theta)
    points[:, 0] = 0.09 + c*along - d*across
    points[:, 1] = d*along + c*across
    points[:, 2] = mat_top + 0.002 + curve_z + rest[:, 2] * np.cos(theta)
    return points


def z_buckle_start(rest, mat_top):
    """Place a shallow Z fold with a flat tail and no changes to the rest mesh."""
    s, y, offset = rest.T
    start, radius, lap = .266, .005, .012
    bend = np.pi * radius
    first = np.clip((s - start) / radius, 0., np.pi)
    backward = np.clip(s - start - bend, 0., lap)
    second = np.clip((s - start - bend - lap) / radius, 0., np.pi)
    end = start + 2 * bend + lap

    # Clear the first turn before bringing the upper tail back down to the mat.
    bridge = .024
    ramp_radius = .10
    pitch = np.arccos(1. - 2. * radius / ramp_radius)
    ramp_length = ramp_radius * pitch
    down = np.clip((s - end - bridge) / ramp_radius, 0., pitch)
    up = np.clip((s - end - bridge - ramp_length) / ramp_radius, 0., pitch)

    x = (.09 + start + np.minimum(s - start, 0.)
         + radius * (np.sin(first) - np.sin(second)) - backward
         + np.clip(s - end, 0., bridge)
         + ramp_radius * (np.sin(down) + np.sin(pitch) + np.sin(up - pitch))
         + np.maximum(s - end - bridge - 2 * ramp_length, 0.))
    z = (mat_top + .002 + radius * (2. - np.cos(first) - np.cos(second))
         - ramp_radius * (1. - np.cos(down) + np.cos(up - pitch) - np.cos(pitch)))
    theta = first - second - down + up
    return np.column_stack((x - offset * np.sin(theta), y, z + offset * np.cos(theta)))


def settled_buckle_start(rest, asset_name='settled_buckle.npz'):
    """Load the fixed initial cloth shape recorded before any manipulation."""
    expected = sha256(rest.astype('<f8').tobytes()).hexdigest()
    with np.load(Path(__file__).with_name(asset_name), allow_pickle=False) as data:
        points = data['points']
        if data['rest_sha256'].item() != expected or points.shape != rest.shape or not np.isfinite(points).all():
            raise ValueError('Settled shirt asset does not match the task mesh')
    return points


def author_surface(stage, path, rest, faces, points, *, frequent_contacts=False):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    def attributes(prim, values):
        for name, value in values.items():
            attr = prim.GetAttribute(name)
            if not attr or not attr.Set(value):
                raise RuntimeError(f"Cannot author native FEM attribute {name}")

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points.tolist()))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.47, 0.65)])
    prim = mesh.GetPrim()
    for api in ("OmniPhysicsDeformableBodyAPI", "OmniPhysicsSurfaceDeformableSimAPI",
                "PhysxSurfaceDeformableBodyAPI"):
        if not prim.ApplyAPI(api):
            raise RuntimeError(f"Missing native cloth schema {api}")
    edge_a = rest[faces[:, 1]] - rest[faces[:, 0]]
    edge_b = rest[faces[:, 2]] - rest[faces[:, 0]]
    area = np.linalg.norm(np.cross(edge_a, edge_b), axis=1).sum() / 2
    attributes(prim, {
        "omniphysics:restShapePoints": Vt.Vec3fArray(rest.tolist()),
        "omniphysics:restTriVtxIndices": [Gf.Vec3i(*map(int, face)) for face in faces],
        "omniphysics:restBendAnglesDefault": "restShapeDefault",
        "omniphysics:mass": float(area * 0.18),
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:disableGravity": False,
        "physxDeformableBody:solverPositionIterationCount": 32,
    })
    if frequent_contacts:
        attributes(prim, {
            "physxDeformableBody:enableSpeculativeCCD": True,
            "physxDeformableBody:collisionPairUpdateFrequency": 8,
        })
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(0.0008)
    collision.CreateRestOffsetAttr(0.0003)
    material = UsdShade.Material.Define(stage, path + "/Cotton")
    shader = UsdShade.Shader.Define(stage, path + "/Cotton/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.28, .47, .65))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.95)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for api in ("OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI",
                "OmniPhysicsSurfaceDeformableMaterialAPI", "PhysxSurfaceDeformableMaterialAPI"):
        if not material.GetPrim().ApplyAPI(api):
            raise RuntimeError(f"Missing native material schema {api}")
    attributes(material.GetPrim(), {
        "omniphysics:density": 300.0,
        "omniphysics:dynamicFriction": 0.55,
        "omniphysics:youngsModulus": 2e6,
        "omniphysics:poissonsRatio": 0.3,
        "omniphysics:surfaceThickness": 0.0006,
        "omniphysics:surfaceBendStiffness": 1.8e5,
        "physxDeformableMaterial:elasticityDamping": 0.005,
        "physxDeformableMaterial:bendDamping": 0.005,
    })
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim
