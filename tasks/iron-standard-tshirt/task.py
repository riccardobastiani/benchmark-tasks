"""Iron the marked front strip and return the iron to its rest."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from resist_isaac.state import pose_state, tensor
from resist_isaac.usd_assets import create_asset

from .board import BOARD_HEIGHT, board_shapes, drape_on_board
from .cloth import author_surface, settled_buckle_start, shirt_mesh, wrinkled_start, z_buckle_start
from .coverage_tracker import CoverageTracker
from .front_finish import FrontFinish
from .pressing_table import PRESS_HEIGHT, drape_front_lip, pressing_table_shapes
from .short_board import SHORT_BOARD_HEIGHT, drape_on_short_board, short_board_shapes
from .soleplate import OUTLINE, _inside_soleplate, _rotation, measure_rest


CONDITIONS = ("recoverable", "diagonal_probe", "diagonal_refined", "diagonal_contact",
              "raised_board", "z_buckle", "settled_buckle", "wide_buckle", "turnover",
              "short_turnover", "settled_short_turnover", "pressing_table")
TABLE_TOP = 0.7475
MAT_TOP = TABLE_TOP + 0.008
REST = (0.20, -0.57, TABLE_TOP + 0.006)
REST_HALF_XY = (.145, .069)


def box(half_extents, position, color, friction=(0.6, 0.5)):
    return dict(type="box", size=half_extents, position=position, color=(*color, 1.), friction=friction)


def iron_description(powered=False):
    # Rounded heel and tapered toe, pointing forward along +X. Sole bottom is z=0.
    def hull(scale, bottom, top, color, friction):
        n = len(OUTLINE)
        points = [(x * scale, y * scale, z) for z in (bottom, top) for x, y in OUTLINE]
        faces = [list(reversed(range(n))), list(range(n, 2*n))]
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j+n, i+n])
        return dict(type="mesh", mesh=dict(points=points, faces=faces), color=(*color, 1.),
                    contact_offset=.001, friction=friction)

    blue = (.18, .30, .42)
    shapes = [hull(1., 0., .009, (.65, .68, .70), (.30, .24)),
              hull(.91, .009, .038, (.82, .86, .88), (.5, .4)),
              box((.012, .023, .032), (-.083, 0., .063), blue),
              box((.014, .021, .026), (.068, 0., .061), blue),
              box((.086, .014, .014), (-.006, 0., .100), blue, (.85, .7))]
    if powered:
        shapes.append(dict(box((.009, .006, .001), (-.070, .036, .039), (.95, .27, .03)), collision=False))
    return dict(name="iron", bodies=[dict(name="iron", mass=1., center_of_mass=(0., 0., .024),
                inertia=dict(diagonal=(.0015, .0065, .0068)), shapes=shapes)])


class IronShirt:
    physics_hz = 240
    hold_s = .05  # FrontFinish itself requires twenty consecutive completed controls.
    robot_contact_offset = .0005

    def __init__(self, condition):
        self.powered = condition == 'recoverable'
        if self.powered:
            condition = 'settled_buckle'
        self.raised_board = condition in ('raised_board', 'turnover')
        self.short_board = condition in ('short_turnover', 'settled_short_turnover')
        self.pressing_table = condition == 'pressing_table'
        self.disable_fabric = os.environ.get('ROBOCURVE_TSHIRT_USD_SYNC') == '1'
        self.render_kit_args = ('--/rtx/hydra/TBNFrameMode=1'
                                if os.environ.get('ROBOCURVE_TSHIRT_CPU_NORMALS') == '1' else '')
        render_gpu = os.environ.get('ROBOCURVE_TSHIRT_RENDER_GPU')
        if render_gpu is not None:
            index = int(render_gpu)
            if index < 0:
                raise ValueError('Render GPU index must be nonnegative')
            self.render_kit_args += f' --/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false'
        self.frequent_contacts = condition == 'diagonal_contact'
        if self.frequent_contacts:
            self.physics_hz = 480
        self.support_top = (TABLE_TOP + PRESS_HEIGHT if self.pressing_table else
                            TABLE_TOP + SHORT_BOARD_HEIGHT if self.short_board else
                            TABLE_TOP + BOARD_HEIGHT if self.raised_board else MAT_TOP)
        torso_wrinkle = self.short_board or self.pressing_table or condition in ('z_buckle', 'settled_buckle', 'wide_buckle', 'turnover')
        self.rest, self.faces, self.n_face = shirt_mesh(refine_corner=condition == 'diagonal_refined',
                                                      refine_z_buckle=torso_wrinkle)
        if condition == 'settled_short_turnover':
            self.initial = settled_buckle_start(self.rest, 'settled_short_board.npz')
        elif condition in ('settled_buckle', 'wide_buckle', 'short_turnover', 'pressing_table'):
            self.initial = settled_buckle_start(self.rest)
            if condition == 'wide_buckle':
                self.initial[:, 1] -= .085
        elif torso_wrinkle:
            self.initial = z_buckle_start(self.rest, MAT_TOP)
        else:
            fold_angle = {'recoverable': 0., 'raised_board': -12.}.get(condition, 12.)
            self.initial = wrinkled_start(self.rest, MAT_TOP, np.deg2rad(fold_angle))
        if self.raised_board:
            self.initial = drape_on_board(self.initial, MAT_TOP, TABLE_TOP)
        if condition == 'short_turnover':
            self.initial = drape_on_short_board(self.initial, MAT_TOP, TABLE_TOP)
        if self.pressing_table:
            self.initial = drape_front_lip(self.initial, MAT_TOP, TABLE_TOP)
        tri = self.rest[self.faces[:self.n_face]]
        self.panel_weights = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2]
        self.flat_double_area = float(self.panel_weights.sum())
        self.temporary = None
        self.coverage = None
        self.track_coverage = condition in ('settled_buckle', 'wide_buckle', 'pressing_table')

    def configure_native(self, cfg):
        from isaaclab_physx.physics import PhysxCfg

        if cfg.sim.physics is None:
            cfg.sim.physics = PhysxCfg()
        if not isinstance(cfg.sim.physics, PhysxCfg):
            raise RuntimeError("T-shirt prototype requires native PhysX")
        cfg.sim.physics.enable_ccd = False
        cfg.sim.physics.solve_articulation_contact_last = True
        for name in ('left_cam', 'right_cam'):
            camera = getattr(cfg.scene, name)
            camera.width, camera.height = 320, 180

    def build_scene(self, embodiment):
        from isaaclab.assets import DeformableObjectCfg
        from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_arena.assets.asset import Asset
        from isaaclab_arena.assets.object import Object, ObjectType
        from isaaclab_arena.utils.pose import Pose

        def spawn(path, cfg, translation=None, orientation=None):
            from pxr import PhysxSchema, UsdPhysics

            stage = get_current_stage()
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    api.CreateEnableGPUDynamicsAttr(True)
                    api.CreateBroadphaseTypeAttr("GPU")
                    api.CreateEnableCCDAttr(False)
            points = self.initial + np.asarray((0., 0., 0.) if translation is None else translation)
            return author_surface(stage, str(path), self.rest, self.faces, points,
                                  frequent_contacts=self.frequent_contacts)

        def finger_pad_setup(env, env_ids):
            self.configure_finger_pads(env, env_ids)
            if self.powered:
                self.create_power_cord()

        class SurfaceShirt(Asset):
            def get_object_cfg(self):
                return self.name, DeformableObjectCfg(prim_path="{ENV_REGEX_NS}/shirt",
                                                      spawn=SpawnerCfg(func=spawn))

            def get_event_cfg(self):
                from isaaclab.managers import EventTermCfg

                return self.name, EventTermCfg(func=finger_pad_setup, mode="prestartup")

        self.temporary = TemporaryDirectory(prefix="iron-standard-tshirt-")
        root = Path(self.temporary.name)
        if self.pressing_table:
            support = pressing_table_shapes(TABLE_TOP)
        elif self.short_board:
            support = short_board_shapes(TABLE_TOP)
        elif self.raised_board:
            support = board_shapes(TABLE_TOP)
        else:
            support = [box((.40, .50, .004), (.42, 0., TABLE_TOP+.004), (.43, .44, .42), (.65, .55))]
        support.append(box((*REST_HALF_XY, .003), (REST[0], REST[1], REST[2]-.003), (.18, .24, .23), (.8, .65)))
        surface = dict(name="support", bodies=[dict(name="support", static=True, shapes=support)])
        return [Object(name="ironing_mat", object_type=ObjectType.BASE,
                       usd_path=str(create_asset(surface, root / "support.usda"))),
                Object(name="iron", object_type=ObjectType.RIGID,
                       usd_path=str(create_asset(iron_description(self.powered), root / "iron.usda")),
                       initial_pose=Pose(position_xyz=REST)),
                SurfaceShirt("shirt", tags=["surface-deformable", "cotton"])]

    def create_power_cord(self):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom, UsdLux

        stage = get_current_stage()
        # A fixed low light lets the same camera reveal the ridge before and after.
        light = UsdLux.SphereLight.Define(stage, "/World/IroningGrazingLight")
        light.CreateIntensityAttr(350.)
        light.CreateRadiusAttr(.05)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(.78, .32, TABLE_TOP+.15))
        outlet = UsdGeom.Cube.Define(stage, "/World/IronOutlet")
        outlet.CreateSizeAttr(1.)
        outlet.AddTranslateOp().Set(Gf.Vec3d(.54, -.633, TABLE_TOP+.018))
        outlet.AddScaleOp().Set(Gf.Vec3d(.075, .030, .036))
        outlet.CreateDisplayColorAttr([(0.25, 0.25, 0.25)])
        self.cord = UsdGeom.BasisCurves.Define(stage, "/World/IronPowerCord")
        self.cord.CreateTypeAttr("linear")
        self.cord.CreateCurveVertexCountsAttr([25])
        self.cord.CreateWidthsAttr([.007])
        self.cord.SetWidthsInterpolation("constant")
        self.cord.CreateDisplayColorAttr([(0.035, 0.035, 0.035)])
        self.update_power_cord(np.array(REST), np.array([0., 0., 0., 1.]))

    def update_power_cord(self, position, orientation):
        from pxr import Vt

        heel = position + _rotation(orientation) @ [-.110, 0., .032]
        bend = heel + _rotation(orientation) @ [-.07, 0., .005]
        guide = np.array([.12, -.618, TABLE_TOP+.010])
        plug = np.array([.505, -.633, TABLE_TOP+.026])
        t = np.linspace(0., 1., 25)[:, None]
        points = (1-t)**3*heel + 3*(1-t)**2*t*bend + 3*(1-t)*t*t*guide + t**3*plug
        self.cord.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))

    def configure_finger_pads(self, env, env_ids):
        from pxr import Usd, UsdPhysics, UsdShade
        from isaaclab.sim.utils.stage import get_current_stage

        # Both arms exist, but PhysX has not cooked their shapes yet. Runtime USD
        # bindings do not update already-cooked articulation shape materials.
        stage = get_current_stage()
        pad_material = UsdShade.Material.Define(stage, "/World/Materials/IroningFingerPads")
        material_api = UsdPhysics.MaterialAPI.Apply(pad_material.GetPrim())
        material_api.CreateStaticFrictionAttr(1.0)
        material_api.CreateDynamicFrictionAttr(1.0)
        pad_names = {"tip_left_009", "tip_left_010", "tip_left_014", "tip_left_015",
                     "tip_right_003", "tip_right_005", "tip_right_008", "tip_right_010"}
        count = 0
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if prim.GetName() in pad_names and prim.HasAPI(UsdPhysics.CollisionAPI):
                binding_prim = prim
                while binding_prim.IsInstanceProxy():
                    binding_prim = binding_prim.GetParent()
                UsdShade.MaterialBindingAPI.Apply(binding_prim).Bind(
                    pad_material, UsdShade.Tokens.strongerThanDescendants, "physics")
                bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if bound.GetPath() != pad_material.GetPath():
                    raise RuntimeError(f"Finger pad material did not bind at {prim.GetPath()}")
                count += 1
        if count != 16:
            raise RuntimeError(f"Expected 16 pinned YAM pad colliders, found {count}")
        self.finger_pad_material_colliders = count

    def reset(self, base):
        import torch

        asset = base.scene["shirt"]
        origin = tensor(base.scene.env_origins)
        ids = torch.arange(len(origin), device=origin.device)
        state = torch.zeros((len(ids), len(self.initial), 6), dtype=origin.dtype, device=origin.device)
        state[:, :, :3] = torch.as_tensor(self.initial, dtype=state.dtype, device=state.device) + origin[:, None, :]
        asset.write_nodal_state_to_sim_index(state, env_ids=ids)
        asset.reset(env_ids=ids)

    def initial_action(self):
        from yam_arena.constants import READY

        return np.r_[READY, 1., READY, 1.]

    def begin_episode(self, base):
        if not self.track_coverage:
            return
        from yam_arena.constants import FINGER_JOINTS

        self.coverage = CoverageTracker(self.rest, (self.faces[:self.n_face], self.faces[self.n_face:2*self.n_face]))
        self.finish = FrontFinish(self.rest, self.faces[:self.n_face], pass_based=self.powered)
        self.pressed = np.zeros(5, dtype=bool)
        self.contact_frames = np.zeros(5, dtype=int)
        self._coverage_step = -1
        self._coverage_arms = {}
        for side in ('left', 'right'):
            arm = base.scene[f'{side}_robot']
            self._coverage_arms[side] = (arm, arm.find_joints(FINGER_JOINTS, preserve_order=True)[0],
                                         arm.backend_body_names.index('gripper'))

    @staticmethod
    def gripper_wrench(arm, link):
        import warp as wp
        from yam_arena.urdf import quat_xyzw_to_matrix

        raw = wp.to_torch(arm.root_view.get_link_incoming_joint_force())
        wrench = raw.detach().cpu().numpy().reshape(len(arm.backend_body_names), 6)[link]
        _, quaternion, _ = pose_state(arm, 'gripper')
        force = quat_xyzw_to_matrix(quaternion.cpu().numpy()) @ wrench[:3]
        return wrench, force

    def observe_coverage(self, base, points, position, orientation):
        from yam_arena.constants import FINGER_OPEN
        from yam_arena.runtime import tcp_world

        step = int(base.common_step_counter)
        if step == self._coverage_step:
            return
        tcp, opening, wrist_z = {}, {}, {}
        for side, (arm, fingers, link) in self._coverage_arms.items():
            tcp[side] = tcp_world(arm)
            opening[side] = float(tensor(arm.data.joint_pos)[0, fingers].mean()) / FINGER_OPEN
            _, force = self.gripper_wrench(arm, link)
            wrist_z[side] = float(force[2])
        self.coverage.observe(step, points, tcp_m=tcp, opening=opening, wrist_force_z_N=wrist_z,
                              iron_position_m=position, iron_orientation_native=orientation)
        self._coverage_step = step

    iron_rest = REST

    def nodes(self, base):
        data = base.scene["shirt"].data
        p = tensor(data.nodal_pos_w)[0, :len(self.rest)].detach().cpu().numpy().copy()
        v = tensor(data.nodal_vel_w)[0, :len(self.rest)].detach().cpu().numpy().copy()
        if p.shape != self.rest.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise RuntimeError("Native cloth telemetry is invalid")
        return p, v

    def geometry(self, points):
        # Measure garment interiors, excluding the stitched edge triangles.
        rows = {}
        for i, face in enumerate(("front", "back")):
            inds = self.faces[i*self.n_face:(i+1)*self.n_face]
            tri = points[inds]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            norm = np.maximum(np.linalg.norm(normals, axis=1), 1e-15)
            slope = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]) / norm, 0, 1)))
            ids = np.unique(inds)
            heights = points[ids, 2] - self.support_top
            rows[face + "_slope_p95_deg"] = float(np.percentile(slope, 95))
            rows[face + "_ridge_p99_mm"] = float(np.percentile(heights, 99) * 1000)
            rows[face + "_steep_fraction"] = float(np.mean(slope > 20))
            rows[face + "_signed_projected_area_ratio"] = float(abs(normals[:, 2].sum()) / self.flat_double_area)
            positive = self.panel_weights[normals[:, 2] > 0].sum()
            negative = self.panel_weights[normals[:, 2] < 0].sum()
            rows[face + "_outward_up_fraction"] = float(positive / self.flat_double_area)
            # A folded flap can have almost zero slope while facing the wrong way.
            rows[face + "_reversed_fraction"] = float(min(positive, negative) / self.flat_double_area)
        return rows

    def evaluate(self, base):
        from yam_arena.runtime import tcp_world

        points, speed = self.nodes(base)
        pos, orientation, _ = pose_state(base.scene["iron"])
        position = pos.detach().cpu().numpy()
        rotation = orientation.detach().cpu().numpy()
        rest = measure_rest(position, rotation, REST, REST_HALF_XY)
        coverage = {}
        step = int(base.common_step_counter)
        if self.powered:
            if self._coverage_step != step:
                self.update_power_cord(position, rotation)
                stations = np.asarray(self.finish.rows)[:, 8]
                local = (points[stations]-position) @ _rotation(rotation)
                front_exposed = points[stations, 2] > points[stations+len(points)//2, 2]
                covered = (_inside_soleplate(local[:, :2]) & front_exposed
                           & (-.0015 <= local[:, 2]) & (local[:, 2] <= .0005)
                           & (_rotation(rotation)[2, 2] >= np.cos(np.deg2rad(5))))
                self.contact_frames = np.where(covered, self.contact_frames+1, 0)
                self.pressed |= self.contact_frames >= 3
                self._coverage_step = step
            coverage = {'pressed_stations': self.pressed}
        elif self.coverage is not None:
            self.observe_coverage(base, points, position, rotation)
            coverage = self.coverage.diagnostics()
        geometry = self.geometry(points)
        finish = dict(success=False, score=0.)
        if self.coverage is not None:
            finish = self.finish.observe(int(base.common_step_counter), points, geometry=geometry,
                coverage=coverage, rest_metrics=rest,
                tcp_m={side: np.asarray(tcp_world(arm)) for side, (arm, _, _) in self._coverage_arms.items()},
                iron_position=position, iron_orientation=rotation)
        coverage.pop("pressed_stations", None)
        return dict(**finish, **geometry,
                    cloth_speed_p95_m_s=float(np.percentile(np.linalg.norm(speed, axis=1), 95)),
                    iron_park_error_m=float(np.linalg.norm(position - REST)), **rest, **coverage)

    def close(self):
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None


def make_task(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}")
    return IronShirt(condition)
