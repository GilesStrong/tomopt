from typing import Tuple, List, Type

from torch import Tensor, nn
import torch.nn.functional as F

from ...volume import Volume
from ...inference.volume import AbsIntClassifierFromX0, AbsX0Inferer

__all__ = ["LadleFurnaceFillLevelInferrer"]


class LadleFurnaceFillLevelInferrer(AbsIntClassifierFromX0):
    r"""
    Research tested only: no unit tests
    """

    def __init__(
        self,
        partial_x0_inferer: Type[AbsX0Inferer],
        volume: Volume,
        pipeline: List[str] = ["remove_ladle", "avg_3d", "avg_layers", "avg_1d", "ridge_1d_0", "negative", "max_div_min"],
        add_batch_dim: bool = True,
        output_probs: bool = True,
    ):
        super().__init__(partial_x0_inferer=partial_x0_inferer, volume=volume, output_probs=output_probs, class2float=self._class2float)
        self.pipeline, self.add_batch_dim = pipeline, add_batch_dim

    @staticmethod
    def _class2float(preds: Tensor, volume: Volume) -> Tensor:
        return ((preds + 1) * volume.passive_size) + volume.get_passive_z_range()[0]

    @staticmethod
    def avg_3d(x: Tensor) -> Tensor:
        return F.avg_pool3d(x, kernel_size=3, padding=1, stride=1, count_include_pad=False)

    @staticmethod
    def gauss_3d(x: Tensor) -> Tensor:
        gauss = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        gauss.weight.data = Tensor([[[[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[2, 4, 2], [4, 8, 4], [2, 4, 2]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]])
        gauss.requires_grad_(False)
        return gauss(x[:, None]).squeeze() / gauss.weight.sum()

    @staticmethod
    def avg_layers(x: Tensor) -> Tensor:
        return x.mean((-1, -2))

    @staticmethod
    def max_sub_min(x: Tensor) -> Tensor:
        maxes = F.max_pool1d(x, kernel_size=3, padding=1, stride=1)
        mins = -F.max_pool1d(-x, kernel_size=3, padding=1, stride=1)
        return maxes - mins

    @staticmethod
    def max_div_min(x: Tensor) -> Tensor:
        maxes = F.max_pool1d(x, kernel_size=3, padding=1, stride=1)
        mins = -F.max_pool1d(-x, kernel_size=3, padding=1, stride=1)
        return maxes / mins

    @staticmethod
    def edge_det(x: Tensor, kernel: Tuple[float, float, float]) -> Tensor:
        edge = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        edge.weight.data = Tensor([[kernel]])
        edge.requires_grad_(False)
        return edge(x[:, None]).squeeze(1)

    def ridge_1d_0(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 0, -1))

    def ridge_1d_2(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 2, -1))

    def ridge_1d_4(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 4, -1))

    def ridge_1d_8(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 8, -1))

    def prewit_1d(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (-1, 0, 1))

    def laplacian_1d(self, x: Tensor) -> Tensor:
        return self.edge_det(x, (1, -4, 1))

    @staticmethod
    def gauss_1d(x: Tensor) -> Tensor:
        gauss = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False)
        gauss.weight.data = Tensor([[[2, 4, 2]]])
        gauss.requires_grad_(False)
        return gauss(x[:, None]).squeeze() / 8

    @staticmethod
    def avg_1d(x: Tensor) -> Tensor:
        return F.avg_pool1d(x, kernel_size=3, padding=1, stride=1, count_include_pad=False)

    @staticmethod
    def negative(x: Tensor) -> Tensor:
        return -x

    @staticmethod
    def remove_ladle(x: Tensor) -> Tensor:
        """Assumes ladle is 1 voxel thick"""
        return x[:, 1:, 1:-1, 1:-1]

    def x02probs(self, vox_preds: Tensor, vox_inv_weights: Tensor) -> Tensor:
        """Can we inlcude the vox_inv_weights? e.g. weighted average?"""
        if self.add_batch_dim:
            vox_preds = vox_preds[None]
        for f in self.pipeline:
            vox_preds = self.__getattribute__(f)(vox_preds)
        if self.add_batch_dim:
            vox_preds = vox_preds[0]
        return F.softmax(vox_preds, dim=-1)
