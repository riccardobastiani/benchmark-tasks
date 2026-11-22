"""Native USD authoring for the task-owned geometry descriptions.

create_asset(description, output_path, source_dir=None) writes one default-prim
asset. Body positions/orientations are absolute in asset coordinates; shape
poses are body-local. Quaternions are scalar first. Primitive sizes are half
extents, or [radius, half_length]. Joint limits and spring constants use SI;
angular values are converted to USD degrees. Drives have fixed passive targets.

A description has name, bodies, optional joints, articulation and fixed_base.
Optional solver_position_iterations/solver_velocity_iterations apply to its rigid bodies.
Bodies provide name, position, orientation, mass, center_of_mass, inertia and
shapes. A body with static=True has collisions but no rigid-body dynamics.
Inertia accepts diagonal/orientation or a full symmetric tensor. Shapes accept
type, size, mesh(points/faces), mesh_file(OBJ/STL), scale, position, orientation,
fromto, color, texture, collision, visible, friction(static/dynamic) and contact
offsets. Joints accept name, type, parent, child, position_parent/child,
orientation_parent/child, axis, lower, upper, stiffness, damping and target.
Spherical joints may declare locked_axes and an angular_limit in radians for free axes.
Single-axis articulation joints may also declare static_friction_effort and
dynamic_friction_effort in N or N m, requiring the registered PhysxJointAxisAPI.

This authors geometry and passive constraints only; it does not run simulation,
convert MJCF, drive task joints or validate physical acceptance.
"""

import math
from pathlib import Path
import re
import struct


def read_mesh(path):
    """Read retained OBJ/STL vertices, polygons and optional face-varying UVs."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        points, texcoords, faces, face_uvs, materials = [], [], [], [], []
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                points.append([float(v) for v in fields[1:4]])
            elif fields[0] == "vt":
                texcoords.append([float(v) for v in fields[1:3]])
            elif fields[0] == "mtllib":
                materials.append(path.parent / " ".join(fields[1:]))
            elif fields[0] == "f":
                face, uv = [], []
                for token in fields[1:]:
                    indices = token.split("/")
                    vertex = int(indices[0])
                    face.append(vertex - 1 if vertex > 0 else len(points) + vertex)
                    if len(indices) > 1 and indices[1]:
                        texture = int(indices[1])
                        uv.append(texture - 1 if texture > 0 else len(texcoords) + texture)
                faces.append(face)
                face_uvs.append(uv)
        result = {"points": points, "faces": faces}
        if texcoords and all(len(f) == len(uv) for f, uv in zip(faces, face_uvs)):
            result["uv"] = [texcoords[i] for face in face_uvs for i in face]
        for material in materials:
            if material.is_file():
                for line in material.read_text().splitlines():
                    if line.strip().startswith("map_Kd "):
                        result["texture"] = str((material.parent / line.strip()[7:]).resolve())
                        return result
        return result
    if path.suffix.lower() == ".stl":
        contents = path.read_bytes()
        count = struct.unpack_from("<I", contents, 80)[0] if len(contents) >= 84 else 0
        if count and len(contents) == 84 + count * 50:
            points = []
            for index in range(count):
                values = struct.unpack_from("<12fH", contents, 84 + index * 50)
                points.extend([list(values[start:start + 3]) for start in (3, 6, 9)])
        else:
            points = [[float(v) for v in line.split()[1:4]] for line in contents.decode().splitlines() if line.strip().startswith("vertex ")]
        return {"points": points, "faces": [list(range(i, i + 3)) for i in range(0, len(points), 3)]}
    raise ValueError(f"Expected retained OBJ or STL mesh: {path}")


def create_asset(description, output_path, source_dir=None):
    """Author rigid parts, real colliders and passive joints without starting Kit."""
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source_dir = Path(source_dir or ".").resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def identifier(value):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"Invalid USD identifier: {value!r}")
        return value

    def vector(values):
        values = list(map(float, values))
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("Expected three finite spatial values")
        return Gf.Vec3f(*values)

    def quaternion(values):
        values = list(map(float, values))
        norm = math.sqrt(sum(v * v for v in values))
        if len(values) != 4 or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Expected a finite nonzero scalar-first quaternion")
        return Gf.Quatf(values[0] / norm, vector(v / norm for v in values[1:]))

    root_path = "/" + identifier(description["name"])
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    body_paths = {}
    for body in description["bodies"]:
        name = identifier(body["name"])
        if name in body_paths:
            raise ValueError(f"Duplicate body: {name}")
        path = root_path + "/" + name
        body_paths[name] = path
        transform = UsdGeom.Xform.Define(stage, path)
        transform.AddTranslateOp().Set(Gf.Vec3d(*body.get("position", [0, 0, 0])))
        transform.AddOrientOp().Set(quaternion(body.get("orientation", [1, 0, 0, 0])))
        if not body.get("static", False):
            UsdPhysics.RigidBodyAPI.Apply(transform.GetPrim())
            if "solver_position_iterations" in description:
                physics = PhysxSchema.PhysxRigidBodyAPI.Apply(transform.GetPrim())
                physics.CreateSolverPositionIterationCountAttr(description["solver_position_iterations"])
                physics.CreateSolverVelocityIterationCountAttr(description["solver_velocity_iterations"])
            mass = UsdPhysics.MassAPI.Apply(transform.GetPrim())
            value = float(body.get("mass", 1.0))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Body mass must be finite and positive: {name}")
            mass.CreateMassAttr(value)
            mass.CreateCenterOfMassAttr(vector(body.get("center_of_mass", [0, 0, 0])))
            inertia = body.get("inertia", {"diagonal": [0.001] * 3})
            if "tensor" in inertia:
                tensor = np.asarray(inertia["tensor"], dtype=float)
                if tensor.shape != (3, 3) or not np.isfinite(tensor).all() or not np.allclose(tensor, tensor.T):
                    raise ValueError(f"Inertia must be a finite symmetric tensor: {name}")
                moments, axes = np.linalg.eigh(tensor)
                if np.linalg.det(axes) < 0:
                    axes[:, 0] *= -1
                rotation = Gf.Matrix3d(*axes.T.ravel().tolist()).ExtractRotation().GetQuat()
                principal = Gf.Quatf(float(rotation.GetReal()), vector(rotation.GetImaginary()))
            else:
                moments = np.asarray(inertia["diagonal"], dtype=float)
                principal = quaternion(inertia.get("orientation", [1, 0, 0, 0]))
            if moments.shape != (3,) or not np.isfinite(moments).all() or np.any(moments <= 0):
                raise ValueError(f"Principal inertia must be finite and positive: {name}")
            mass.CreateDiagonalInertiaAttr(vector(moments))
            mass.CreatePrincipalAxesAttr(principal)

        for index, shape in enumerate(body.get("shapes", [])):
            shape_path = f"{path}/shape_{index:03d}"
            kind = shape["type"]
            size = shape.get("size", [])
            mesh = None
            if kind == "box":
                geometry = UsdGeom.Cube.Define(stage, shape_path)
                geometry.CreateSizeAttr(1.0)
            elif kind in ("cylinder", "capsule"):
                schema = UsdGeom.Cylinder if kind == "cylinder" else UsdGeom.Capsule
                geometry = schema.Define(stage, shape_path)
                geometry.CreateRadiusAttr(float(size[0]))
                height = 2 * float(size[1]) if len(size) > 1 else 0.0
                if "fromto" in shape:
                    a, b = shape["fromto"][:3], shape["fromto"][3:]
                    height = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
                geometry.CreateHeightAttr(height)
                geometry.CreateAxisAttr(UsdGeom.Tokens.z)
            elif kind == "sphere":
                geometry = UsdGeom.Sphere.Define(stage, shape_path)
                geometry.CreateRadiusAttr(float(size[0]))
            elif kind == "mesh":
                mesh = shape.get("mesh")
                if mesh is None:
                    mesh = read_mesh(source_dir / shape["mesh_file"])
                points = [vector(p) for p in mesh["points"]]
                faces = mesh["faces"]
                if not points or not faces or any(len(f) < 3 or any(not 0 <= i < len(points) for i in f) for f in faces):
                    raise ValueError(f"Mesh needs valid polygons: {shape.get('name', shape_path)}")
                geometry = UsdGeom.Mesh.Define(stage, shape_path)
                geometry.CreatePointsAttr(points)
                geometry.CreateFaceVertexCountsAttr([len(f) for f in faces])
                geometry.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
                geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                if "uv" in mesh:
                    if len(mesh["uv"]) != sum(len(f) for f in faces):
                        raise ValueError("Mesh UVs must be face-varying")
                    uv = UsdGeom.PrimvarsAPI(geometry).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                    uv.Set([Gf.Vec2f(*map(float, p)) for p in mesh["uv"]])
            else:
                raise ValueError(f"Unsupported native shape: {kind}")
            shape_transform = UsdGeom.Xformable(geometry.GetPrim())
            position = shape.get("position", [0, 0, 0])
            rotation = quaternion(shape.get("orientation", [1, 0, 0, 0]))
            if "fromto" in shape:
                a, b = shape["fromto"][:3], shape["fromto"][3:]
                position = [(float(x) + float(y)) / 2 for x, y in zip(a, b)]
                direction = Gf.Vec3d(*[float(y) - float(x) for x, y in zip(a, b)])
                q = Gf.Rotation(Gf.Vec3d(0, 0, 1), direction).GetQuat()
                rotation = Gf.Quatf(float(q.GetReal()), vector(q.GetImaginary()))
            shape_transform.AddTranslateOp().Set(Gf.Vec3d(*position))
            shape_transform.AddOrientOp().Set(rotation)
            scale = shape.get("scale", [1, 1, 1])
            if kind == "box":
                scale = [2 * float(half) * float(factor) for half, factor in zip(size, scale)]
            shape_transform.AddScaleOp().Set(vector(scale))
            color = shape.get("color", [0.5, 0.5, 0.5, 1.0])
            geometry.CreateDisplayColorAttr([vector(color[:3])])
            geometry.CreateDisplayOpacityAttr([float(color[3])])
            if not shape.get("visible", True):
                UsdGeom.Imageable(geometry.GetPrim()).MakeInvisible()
            material_path = root_path + f"/Materials/{name}_{index:03d}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vector(color[:3]))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(shape.get("roughness", 0.4)))
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            texture = shape.get("texture") or (mesh.get("texture") if mesh else None)
            if texture:
                reader = UsdShade.Shader.Define(stage, material_path + "/Coordinates")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                sampler = UsdShade.Shader.Define(stage, material_path + "/Texture")
                sampler.CreateIdAttr("UsdUVTexture")
                sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str((source_dir / texture).resolve())))
                sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            binding = UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim())
            binding.Bind(material)
            if shape.get("collision", True):
                UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
                if kind == "mesh":
                    UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr(shape.get("collision_approximation", "convexHull"))
                collider = PhysxSchema.PhysxCollisionAPI.Apply(geometry.GetPrim())
                collider.CreateContactOffsetAttr(float(shape.get("contact_offset", 0.0001)))
                collider.CreateRestOffsetAttr(float(shape.get("rest_offset", 0.0)))
                physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                static, dynamic = shape.get("friction", [0.6, 0.5])
                physics_material.CreateStaticFrictionAttr(float(static))
                physics_material.CreateDynamicFrictionAttr(float(dynamic))
                physics_material.CreateRestitutionAttr(0.0)
                binding.Bind(material, materialPurpose="physics")

    if description.get("articulation", False):
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        physics = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        physics.CreateEnabledSelfCollisionsAttr(description.get("self_collisions", True))
        physics.CreateSolverPositionIterationCountAttr(32)
        physics.CreateSolverVelocityIterationCountAttr(8)

    joints = list(description.get("joints", []))
    if description.get("fixed_base"):
        base = next(body for body in description["bodies"] if body["name"] == description["fixed_base"])
        joints.insert(0, {
            "name": "base_fixed", "type": "fixed", "child": base["name"],
            "position_parent": base.get("position", [0, 0, 0]),
            "orientation_parent": base.get("orientation", [1, 0, 0, 0]),
        })
    for definition in joints:
        path = root_path + "/" + identifier(definition["name"])
        kind = definition["type"]
        schemas = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint, "spherical": UsdPhysics.Joint}
        if kind not in schemas:
            raise ValueError(f"Unsupported native joint: {kind}")
        joint = schemas[kind].Define(stage, path)
        if definition.get("parent"):
            joint.CreateBody0Rel().SetTargets([body_paths[definition["parent"]]])
        joint.CreateBody1Rel().SetTargets([body_paths[definition["child"]]])
        joint.CreateLocalPos0Attr(vector(definition.get("position_parent", [0, 0, 0])))
        joint.CreateLocalPos1Attr(vector(definition.get("position_child", [0, 0, 0])))
        joint.CreateLocalRot0Attr(quaternion(definition.get("orientation_parent", [1, 0, 0, 0])))
        joint.CreateLocalRot1Attr(quaternion(definition.get("orientation_child", [1, 0, 0, 0])))
        joint.CreateCollisionEnabledAttr(definition.get("collision", False))
        factor = 180 / math.pi if kind in ("revolute", "spherical") else 1.0
        if kind in ("revolute", "prismatic"):
            joint.CreateAxisAttr(definition.get("axis", "X"))
            if "lower" in definition:
                joint.CreateLowerLimitAttr(float(definition["lower"]) * factor)
            if "upper" in definition:
                joint.CreateUpperLimitAttr(float(definition["upper"]) * factor)
        if kind == "spherical":
            locked = definition.get("locked_axes", [])
            if any(axis not in ("rotX", "rotY", "rotZ") for axis in locked):
                raise ValueError("Spherical joint locks must name rotational axes")
            for axis in ["transX", "transY", "transZ", *locked]:
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
            if "angular_limit" in definition:
                angle = float(definition["angular_limit"])
                if not math.isfinite(angle) or not 0 < angle < math.pi:
                    raise ValueError("Angular limit must be between zero and pi radians")
                for axis in ("rotX", "rotY", "rotZ"):
                    if axis not in locked:
                        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                        limit.CreateLowAttr(-angle * factor)
                        limit.CreateHighAttr(angle * factor)
        if "static_friction_effort" in definition or "dynamic_friction_effort" in definition:
            if not description.get("articulation", False):
                raise ValueError("PhysX joint friction requires an articulation; standalone joints ignore it")
            if kind not in ("revolute", "prismatic"):
                raise ValueError("Effort friction currently requires a single-axis native joint")
            static = float(definition.get("static_friction_effort", 0.0))
            dynamic = float(definition.get("dynamic_friction_effort", static))
            if not all(math.isfinite(v) for v in (static, dynamic)) or not 0 <= dynamic <= static:
                raise ValueError("Joint friction requires finite static >= dynamic >= 0 efforts")
            channel = "angular" if kind == "revolute" else "linear"
            prim = joint.GetPrim()
            if not prim.CanApplyAPI("PhysxJointAxisAPI", channel) or not prim.ApplyAPI("PhysxJointAxisAPI", channel):
                raise RuntimeError("This asset requires the native PhysxJointAxisAPI schema from Isaac Sim 6.1")
            prim.CreateAttribute(f"physxJointAxis:{channel}:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(static)
            prim.CreateAttribute(f"physxJointAxis:{channel}:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(dynamic)
        if any(field in definition for field in ("stiffness", "damping")):
            channels = ("rotX", "rotY", "rotZ") if kind == "spherical" else ("angular" if kind == "revolute" else "linear",)
            for channel in channels:
                if channel in definition.get("locked_axes", []):
                    continue
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), channel)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(float(definition.get("stiffness", 0.0)) / factor)
                drive.CreateDampingAttr(float(definition.get("damping", 0.0)) / factor)
                drive.CreateTargetPositionAttr(float(definition.get("target", 0.0)) * factor)
                drive.CreateTargetVelocityAttr(0.0)
    stage.GetRootLayer().Save()
    return output_path
