"""Gaussian parameter container plus ``.ply`` / ``.npz`` serialisation.

Parameterisation follows the original 3D Gaussian Splatting work (Kerbl et
al. 2023) so that checkpoints interoperate with existing viewers:

======================  ============================================================
attribute               meaning
======================  ============================================================
``means``               (N, 3) centres, **world ENU metres**
``log_scales``          (N, 3) log of the per-axis standard deviations, metres
``quats``               (N, 4) rotation ``(w, x, y, z)``, normalised on use
``sh_dc``               (N, 3) degree-0 spherical harmonic coefficient
``sh_rest``             (N, K, 3) higher SH coefficients, ``K = (deg+1)^2 - 1``
``logit_opacities``     (N,) opacity before the sigmoid
``layer``               (N,) :class:`Layer` id -- losses and exports treat these differently
``t_center``            (N,) optional: centre of the temporal window, project seconds
``t_log_scale``         (N,) optional: log sigma of the temporal window, seconds
======================  ============================================================

The two optional time attributes implement "temporal opacity" in the sense
of Spacetime Gaussians (Li et al. 2024): a gaussian visible only around one
moment, which is exactly what a puff of dust or a falling facade panel is.
They are ignored by static rendering and by third-party viewers.

Units and frames are the project ones throughout: metres in the ENU world
frame of :mod:`wtc4d.world`, seconds since local midnight (see
:mod:`wtc4d.timeline`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = ["Gaussians", "Layer", "SH_C0", "rgb_to_sh_dc", "sh_dc_to_rgb"]

#: Degree-0 spherical harmonic constant, ``0.5 / sqrt(pi)``.
SH_C0 = 0.28209479177387814

_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


class Layer(IntEnum):
    """Semantic layer of a gaussian.

    Losses use this: static layers are supervised only by unmasked pixels,
    ``SMOKE``/``DEBRIS`` only inside dynamic masks and only near their own
    time window.  Exports use it to split a scene into
    :class:`wtc4d.schema.scene.SplatAsset` entries the viewer can toggle.
    """

    TOWERS = 0
    BUILDINGS = 1
    GROUND = 2
    SMOKE = 3
    DEBRIS = 4
    SKY = 5
    OTHER = 6

    @property
    def is_static(self) -> bool:
        return self in (Layer.TOWERS, Layer.BUILDINGS, Layer.GROUND, Layer.SKY)

    @property
    def asset_layer(self) -> str:
        """Matching ``SplatAsset.layer`` string for the viewer manifest."""
        return {
            Layer.TOWERS: "towers",
            Layer.BUILDINGS: "scene",
            Layer.GROUND: "ground",
            Layer.SMOKE: "smoke",
            Layer.DEBRIS: "debris",
            Layer.SKY: "scene",
            Layer.OTHER: "scene",
        }[self]


def rgb_to_sh_dc(rgb: Tensor) -> Tensor:
    """Linear RGB in [0, 1] -> degree-0 SH coefficient."""
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(dc: Tensor) -> Tensor:
    """Degree-0 SH coefficient -> RGB (unclamped)."""
    return SH_C0 * dc + 0.5


def _eval_sh(degree: int, coeffs: Tensor, dirs: Tensor) -> Tensor:
    """Evaluate SH of ``degree`` at unit ``dirs`` (N, 3).  ``coeffs`` is (N, K, 3)."""
    result = SH_C0 * coeffs[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = (
            result
            - _SH_C1 * y * coeffs[:, 1]
            + _SH_C1 * z * coeffs[:, 2]
            - _SH_C1 * x * coeffs[:, 3]
        )
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _SH_C2[0] * xy * coeffs[:, 4]
                + _SH_C2[1] * yz * coeffs[:, 5]
                + _SH_C2[2] * (2.0 * zz - xx - yy) * coeffs[:, 6]
                + _SH_C2[3] * xz * coeffs[:, 7]
                + _SH_C2[4] * (xx - yy) * coeffs[:, 8]
            )
            if degree >= 3:
                result = (
                    result
                    + _SH_C3[0] * y * (3.0 * xx - yy) * coeffs[:, 9]
                    + _SH_C3[1] * xy * z * coeffs[:, 10]
                    + _SH_C3[2] * y * (4.0 * zz - xx - yy) * coeffs[:, 11]
                    + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[:, 12]
                    + _SH_C3[4] * x * (4.0 * zz - xx - yy) * coeffs[:, 13]
                    + _SH_C3[5] * z * (xx - yy) * coeffs[:, 14]
                    + _SH_C3[6] * x * (xx - 3.0 * yy) * coeffs[:, 15]
                )
    return result + 0.5


@dataclass
class Gaussians:
    """A set of 3D gaussians (see the module docstring for the parameterisation)."""

    means: Tensor
    log_scales: Tensor
    quats: Tensor
    sh_dc: Tensor
    logit_opacities: Tensor
    sh_rest: Tensor | None = None
    layer: Tensor | None = None
    t_center: Tensor | None = None
    t_log_scale: Tensor | None = None

    # ---------------------------------------------------------------- basics
    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if self.means.shape != (n, 3):
            raise ValueError("means must be (N, 3)")
        for name, shape in (
            ("log_scales", (n, 3)),
            ("quats", (n, 4)),
            ("sh_dc", (n, 3)),
        ):
            t = getattr(self, name)
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(t.shape)}")
        if self.logit_opacities.shape != (n,):
            if self.logit_opacities.shape == (n, 1):
                self.logit_opacities = self.logit_opacities.squeeze(-1)
            else:
                raise ValueError(
                    f"logit_opacities must be (N,), got {tuple(self.logit_opacities.shape)}"
                )
        if self.sh_rest is not None and self.sh_rest.numel() == 0:
            self.sh_rest = None
        if self.sh_rest is not None and self.sh_rest.shape[0] != n:
            raise ValueError("sh_rest must be (N, K, 3)")
        if self.layer is None:
            self.layer = torch.full((n,), int(Layer.OTHER), dtype=torch.int64, device=self.device)

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def n(self) -> int:
        return len(self)

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def sh_degree(self) -> int:
        if self.sh_rest is None:
            return 0
        return int(round((self.sh_rest.shape[1] + 1) ** 0.5)) - 1

    def tensors(self) -> dict[str, Tensor]:
        return {
            f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None
        }

    def to(self, device=None, dtype=None) -> Gaussians:
        def cast(t: Tensor | None) -> Tensor | None:
            if t is None:
                return None
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        return replace(self, **{k: cast(v) for k, v in self.tensors().items()})

    def detach(self) -> Gaussians:
        return replace(self, **{k: v.detach() for k, v in self.tensors().items()})

    def clone(self) -> Gaussians:
        return replace(self, **{k: v.clone() for k, v in self.tensors().items()})

    def requires_grad_(self, flag: bool = True) -> Gaussians:
        for name in (
            "means",
            "log_scales",
            "quats",
            "sh_dc",
            "sh_rest",
            "logit_opacities",
            "t_center",
            "t_log_scale",
        ):
            t = getattr(self, name)
            if t is not None and t.is_floating_point():
                t.requires_grad_(flag)
        return self

    # ------------------------------------------------------------ activations
    def scales(self) -> Tensor:
        """Per-axis standard deviations in metres, (N, 3)."""
        return torch.exp(self.log_scales)

    def opacities(self) -> Tensor:
        """Opacity in [0, 1], (N,)."""
        return torch.sigmoid(self.logit_opacities)

    def unit_quats(self) -> Tensor:
        return self.quats / self.quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def colors(self, dirs: Tensor | None = None) -> Tensor:
        """RGB in [0, 1] (unclamped), (N, 3).

        With ``dirs`` (unit view directions, world frame, from gaussian to
        camera) the full SH is evaluated; without, only the DC term.
        """
        if self.sh_rest is None or dirs is None:
            return sh_dc_to_rgb(self.sh_dc)
        coeffs = torch.cat([self.sh_dc.unsqueeze(1), self.sh_rest], dim=1)
        return _eval_sh(self.sh_degree, coeffs, dirs)

    def covariances(self) -> Tensor:
        """World-frame 3x3 covariance of every gaussian, (N, 3, 3)."""
        r = quat_to_rotmat_torch(self.unit_quats())
        s = self.scales()
        m = r * s.unsqueeze(-2)  # R @ diag(s)
        return m @ m.transpose(-1, -2)

    def temporal_weight(self, t: float | Tensor) -> Tensor:
        """Gaussian temporal window weight in [0, 1], (N,).

        1 everywhere when the set carries no time attributes.
        """
        if self.t_center is None or self.t_log_scale is None:
            return torch.ones(self.n, device=self.device, dtype=self.means.dtype)
        tt = torch.as_tensor(t, device=self.device, dtype=self.means.dtype)
        sigma = torch.exp(self.t_log_scale).clamp_min(1e-3)
        return torch.exp(-0.5 * ((tt - self.t_center) / sigma) ** 2)

    def opacities_at(self, t: float | Tensor | None) -> Tensor:
        """Opacity in [0, 1] modulated by the temporal window, (N,)."""
        if t is None:
            return self.opacities()
        return self.opacities() * self.temporal_weight(t)

    # -------------------------------------------------------------- selection
    def select(self, index: Tensor) -> Gaussians:
        """Index (bool mask or int index) into a new :class:`Gaussians`."""
        return replace(self, **{k: v[index] for k, v in self.tensors().items()})

    def layer_mask(self, *layers: Layer) -> Tensor:
        assert self.layer is not None
        want = torch.tensor([int(x) for x in layers], device=self.layer.device)
        return torch.isin(self.layer, want)

    @staticmethod
    def cat(parts) -> Gaussians:
        """Concatenate gaussian sets; SH degree is raised to the maximum present."""
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            raise ValueError("nothing to concatenate")
        if len(parts) == 1:
            return parts[0]
        deg = max(p.sh_degree for p in parts)
        k = (deg + 1) ** 2 - 1
        dtype, device = parts[0].means.dtype, parts[0].means.device

        def rest(p: Gaussians) -> Tensor:
            out = torch.zeros(len(p), k, 3, dtype=dtype, device=device)
            if p.sh_rest is not None:
                out[:, : p.sh_rest.shape[1]] = p.sh_rest.to(device=device, dtype=dtype)
            return out

        has_time = any(p.t_center is not None for p in parts)

        def times(p: Gaussians, attr: str, default: float) -> Tensor:
            t = getattr(p, attr)
            if t is None:
                return torch.full((len(p),), default, dtype=dtype, device=device)
            return t.to(device=device, dtype=dtype)

        return Gaussians(
            means=torch.cat([p.means.to(device, dtype) for p in parts]),
            log_scales=torch.cat([p.log_scales.to(device, dtype) for p in parts]),
            quats=torch.cat([p.unit_quats().to(device, dtype) for p in parts]),
            sh_dc=torch.cat([p.sh_dc.to(device, dtype) for p in parts]),
            sh_rest=torch.cat([rest(p) for p in parts]) if k else None,
            logit_opacities=torch.cat([p.logit_opacities.to(device, dtype) for p in parts]),
            layer=torch.cat([p.layer.to(device) for p in parts]),
            t_center=torch.cat([times(p, "t_center", 0.0) for p in parts]) if has_time else None,
            t_log_scale=(
                torch.cat([times(p, "t_log_scale", 6.0) for p in parts]) if has_time else None
            ),
        )

    # ------------------------------------------------------------ constructors
    @staticmethod
    def from_points(
        xyz,
        rgb=None,
        scale_m: float | np.ndarray = 1.0,
        opacity: float = 0.1,
        layer: Layer | np.ndarray = Layer.OTHER,
        sh_degree: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Gaussians:
        """Isotropic, identity-rotation gaussians at ``xyz`` with colour ``rgb``.

        ``scale_m`` is the initial standard deviation in metres (scalar or
        per-point, e.g. from a nearest-neighbour spacing estimate).
        """
        xyz_t = torch.as_tensor(np.asarray(xyz, dtype=np.float64), dtype=dtype, device=device)
        n = xyz_t.shape[0]
        if rgb is None:
            rgb_t = torch.full((n, 3), 0.5, dtype=dtype, device=device)
        else:
            rgb_t = torch.as_tensor(np.asarray(rgb, dtype=np.float64), dtype=dtype, device=device)
            if rgb_t.ndim == 1:
                rgb_t = rgb_t.unsqueeze(0).expand(n, 3).contiguous()
        scale_t = torch.as_tensor(
            np.broadcast_to(np.asarray(scale_m, dtype=np.float64), (n,)).copy(),
            dtype=dtype,
            device=device,
        )
        log_scales = torch.log(scale_t.clamp_min(1e-4)).unsqueeze(-1).repeat(1, 3)
        quats = torch.zeros(n, 4, dtype=dtype, device=device)
        quats[:, 0] = 1.0
        k = (sh_degree + 1) ** 2 - 1
        layer_t = torch.as_tensor(
            np.broadcast_to(np.asarray(layer, dtype=np.int64), (n,)).copy(),
            dtype=torch.int64,
            device=device,
        )
        return Gaussians(
            means=xyz_t,
            log_scales=log_scales,
            quats=quats,
            sh_dc=rgb_to_sh_dc(rgb_t.clamp(0.0, 1.0)),
            sh_rest=torch.zeros(n, k, 3, dtype=dtype, device=device) if k else None,
            logit_opacities=torch.full(
                (n,), float(np.log(opacity / (1.0 - opacity))), dtype=dtype, device=device
            ),
            layer=layer_t,
        )

    def with_sh_degree(self, degree: int) -> Gaussians:
        """Pad or truncate the SH bands to ``degree`` (keeps the DC term)."""
        k = (degree + 1) ** 2 - 1
        if k == 0:
            return replace(self, sh_rest=None)
        rest = torch.zeros(self.n, k, 3, dtype=self.means.dtype, device=self.device)
        if self.sh_rest is not None:
            keep = min(k, self.sh_rest.shape[1])
            rest[:, :keep] = self.sh_rest[:, :keep]
        return replace(self, sh_rest=rest)

    # ----------------------------------------------------------------- ply/npz
    def save_ply(self, path: str | Path) -> Path:
        """Write a standard 3DGS ``.ply`` (binary little endian).

        Field names and ordering match the reference implementation, so
        SuperSplat / nerfstudio / antimatter15-style viewers load it.  Our
        extensions (``layer``, ``t_center``, ``t_log_scale``) are appended as
        extra scalar properties; readers that do not know them ignore them.
        """
        from plyfile import PlyData, PlyElement

        g = self.detach().to(dtype=torch.float32)
        n = g.n
        arrays: dict[str, np.ndarray] = {
            "x": g.means[:, 0].numpy(),
            "y": g.means[:, 1].numpy(),
            "z": g.means[:, 2].numpy(),
            "nx": np.zeros(n, dtype=np.float32),
            "ny": np.zeros(n, dtype=np.float32),
            "nz": np.zeros(n, dtype=np.float32),
        }
        for i in range(3):
            arrays[f"f_dc_{i}"] = g.sh_dc[:, i].numpy()
        if g.sh_rest is not None:
            # Reference layout is channel-major: f_rest_{c * K + k}
            rest = g.sh_rest.permute(0, 2, 1).reshape(n, -1).numpy()
            for i in range(rest.shape[1]):
                arrays[f"f_rest_{i}"] = rest[:, i]
        arrays["opacity"] = g.logit_opacities.numpy()
        for i in range(3):
            arrays[f"scale_{i}"] = g.log_scales[:, i].numpy()
        for i in range(4):
            arrays[f"rot_{i}"] = g.unit_quats()[:, i].numpy()
        arrays["layer"] = g.layer.to(torch.int32).numpy()
        if g.t_center is not None:
            arrays["t_center"] = g.t_center.numpy()
        if g.t_log_scale is not None:
            arrays["t_log_scale"] = g.t_log_scale.numpy()

        dtype = [(k, "i4" if k == "layer" else "f4") for k in arrays]
        data = np.empty(n, dtype=dtype)
        for k, v in arrays.items():
            data[k] = v
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
        return path

    @staticmethod
    def load_ply(
        path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> Gaussians:
        """Read a standard 3DGS ``.ply``.  Missing extensions get defaults."""
        from plyfile import PlyData

        el = PlyData.read(str(path))["vertex"]
        names = {p.name for p in el.properties}

        def col(name: str) -> Tensor:
            return torch.as_tensor(
                np.asarray(el[name], dtype=np.float64), dtype=dtype, device=device
            )

        means = torch.stack([col("x"), col("y"), col("z")], dim=-1)
        n = means.shape[0]
        sh_dc = torch.stack([col(f"f_dc_{i}") for i in range(3)], dim=-1)
        rest_names = sorted(
            (p for p in names if p.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1])
        )
        sh_rest = None
        if rest_names:
            flat = torch.stack([col(p) for p in rest_names], dim=-1)
            k = flat.shape[1] // 3
            sh_rest = flat.reshape(n, 3, k).permute(0, 2, 1).contiguous()
        log_scales = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
        quats = torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1)
        layer = (
            torch.as_tensor(np.asarray(el["layer"], dtype=np.int64), device=device)
            if "layer" in names
            else None
        )
        return Gaussians(
            means=means,
            log_scales=log_scales,
            quats=quats,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            logit_opacities=col("opacity"),
            layer=layer,
            t_center=col("t_center") if "t_center" in names else None,
            t_log_scale=col("t_log_scale") if "t_log_scale" in names else None,
        )

    def save_npz(self, path: str | Path) -> Path:
        """Write a compressed ``.npz`` (our own checkpoint format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        g = self.detach().to(dtype=torch.float32)
        np.savez_compressed(path, **{k: v.numpy() for k, v in g.tensors().items()})
        return path

    @staticmethod
    def load_npz(
        path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> Gaussians:
        with np.load(str(path)) as z:
            data = {k: z[k] for k in z.files}
        kw = {}
        for k, v in data.items():
            kw[k] = torch.as_tensor(v, device=device, dtype=torch.int64 if k == "layer" else dtype)
        return Gaussians(**kw)


def quat_to_rotmat_torch(q: Tensor) -> Tensor:
    """(N, 4) unit quaternion ``(w, x, y, z)`` -> (N, 3, 3) rotation, differentiable."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )
