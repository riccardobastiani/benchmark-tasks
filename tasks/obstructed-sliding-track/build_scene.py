"""Author a glass display and loose L-key as native USD/PhysX assets."""

import argparse
import json
from pathlib import Path


def create_assets(directory):
    """Write a passive door articulation and an independent gravity-loaded key."""
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    geometry = json.loads(Path(__file__).with_name("geometry.json").read_text())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def vec(values):
        return Gf.Vec3f(*map(float, values))

    def shape(stage, root, index, data):
        path = f"{root}/shape_{index:03d}"
        kind = data["type"]
        if kind == "box":
            item = UsdGeom.Cube.Define(stage, path)
            item.CreateSizeAttr(1.0)
        elif kind == "capsule":
            item = UsdGeom.Capsule.Define(stage, path)
            item.CreateRadiusAttr(data["size"][0])
            item.CreateHeightAttr(2 * data["size"][1])
            item.CreateAxisAttr(UsdGeom.Tokens.z)
        elif kind == "mesh":
            source = geometry["meshes"][data["mesh"]]
            item = UsdGeom.Mesh.Define(stage, path)
            item.CreatePointsAttr([vec(p) for p in source["points"]])
            item.CreateFaceVertexCountsAttr([len(face) for face in source["faces"]])
            item.CreateFaceVertexIndicesAttr([i for face in source["faces"] for i in face])
            item.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        else:
            raise ValueError(f"Unsupported display geometry: {kind}")
        transform = UsdGeom.Xformable(item.GetPrim())
        transform.AddTranslateOp().Set(Gf.Vec3d(*data["pos"]))
        if kind == "box":
            transform.AddScaleOp().Set(vec([2 * value for value in data["size"]]))
        asset_root = root.strip("/").split("/")[0]
        material_path = f"/{asset_root}/Materials/{data['material']}"
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            material = UsdShade.Material.Define(stage, material_path)
            rgba = geometry["materials"][data["material"]]
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vec(rgba[:3]))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(rgba[3])
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15 if data["material"] == "s_glass" else 0.4)
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(0.6)
            physics.CreateDynamicFrictionAttr(0.5)
            physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(item.GetPrim()).Bind(material)
        UsdShade.MaterialBindingAPI(item.GetPrim()).Bind(material, materialPurpose="physics")
        UsdPhysics.CollisionAPI.Apply(item.GetPrim())
        if kind == "mesh":
            UsdPhysics.MeshCollisionAPI.Apply(item.GetPrim()).CreateApproximationAttr("convexHull")
        contacts = PhysxSchema.PhysxCollisionAPI.Apply(item.GetPrim())
        contacts.CreateContactOffsetAttr(0.0001)
        contacts.CreateRestOffsetAttr(0.0)

    def stage_at(path, root):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        prim = UsdGeom.Xform.Define(stage, root)
        stage.SetDefaultPrim(prim.GetPrim())
        return stage

    display_path = directory / "display.usda"
    stage = stage_at(display_path, "/DisplayCase")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetDefaultPrim())
    articulation.CreateEnabledSelfCollisionsAttr(True)
    articulation.CreateSolverPositionIterationCountAttr(32)
    articulation.CreateSolverVelocityIterationCountAttr(8)
    for name, offset in (("Frame", [0, 0, 0]), ("Door", geometry["door_local_position_m"])):
        body = UsdGeom.Xform.Define(stage, f"/DisplayCase/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*offset))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        if name == "Door":
            source = geometry["door_inertia"]
            moments, axes = np.linalg.eigh(np.asarray(source["tensor"]))
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            rotation = Gf.Matrix3d(*axes.T.ravel().tolist()).ExtractRotation().GetQuat()
            mass.CreateMassAttr(source["mass"])
            mass.CreateCenterOfMassAttr(vec(source["center_of_mass"]))
            mass.CreateDiagonalInertiaAttr(vec(moments))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(float(rotation.GetReal()), vec(rotation.GetImaginary())))
        else:
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr(vec([0.1, 0.1, 0.1]))
        for index, item in enumerate(geometry[name.lower()]):
            shape(stage, str(body.GetPath()), index, item)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/DisplayCase/base_fixed")
    fixed.CreateBody1Rel().SetTargets(["/DisplayCase/Frame"])
    slider = UsdPhysics.PrismaticJoint.Define(stage, "/DisplayCase/door_slide")
    slider.CreateBody0Rel().SetTargets(["/DisplayCase/Frame"])
    slider.CreateBody1Rel().SetTargets(["/DisplayCase/Door"])
    slider.CreateAxisAttr("Y")
    slider.CreateLocalPos0Attr(vec(geometry["door_local_position_m"]))
    slider.CreateLocalPos1Attr(vec([0, 0, 0]))
    slider.CreateLowerLimitAttr(0.0)
    slider.CreateUpperLimitAttr(0.29)
    slider.CreateCollisionEnabledAttr(True)
    damping = UsdPhysics.DriveAPI.Apply(slider.GetPrim(), "linear")
    damping.CreateTypeAttr("force")
    damping.CreateStiffnessAttr(0.0)
    damping.CreateDampingAttr(1.0)
    stage.GetRootLayer().Save()

    key_path = directory / "track_key.usda"
    stage = stage_at(key_path, "/TrackKey")
    key = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(key)
    mass = UsdPhysics.MassAPI.Apply(key)
    mass.CreateMassAttr(geometry["key_inertia"]["mass"])
    mass.CreateCenterOfMassAttr(vec(geometry["key_inertia"]["center_of_mass"]))
    mass.CreateDiagonalInertiaAttr(vec(geometry["key_inertia"]["diagonal"]))
    for index, item in enumerate(geometry["key"]):
        shape(stage, "/TrackKey", index, item)
    stage.GetRootLayer().Save()
    return {"display": display_path, "track_key": key_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for name, path in create_assets(args.directory).items():
        print(f"{name}: {path}")
