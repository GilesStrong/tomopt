from typing import Dict, Optional, Union, Callable
from abc import abstractmethod, ABCMeta

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .sub_losses import integer_class_loss
from ...volume import Volume

r"""
Provides loss functions for evaluating the performance of detector and inference configurations
"""

__all__ = ["AbsDetectorLoss", "AbsMaterialClassLoss", "VoxelX0Loss", "VoxelClassLoss", "VolumeClassLoss", "VolumeIntClassLoss"]


class AbsDetectorLoss(nn.Module, metaclass=ABCMeta):
    r"""
    Abstract base class from which all loss functions should inherit.

    The loss consists of:
        - A component that quantifies the performance of the predictions made via the detectors
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The performance component (error) should ideally be as close to the final task that the detector will be performing,
    and will depend on the output of the inference algorithm used

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Inheriting classes will need to at least override the `_get_inference_loss` method.

    Arguments:
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def __init__(
        self,
        *,
        target_budget: Optional[float],
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.target_budget = target_budget
        self.budget_smoothing = budget_smoothing
        self.cost_coef = cost_coef
        self.steep_budget = steep_budget
        self.debug = debug
        self.sub_losses: Dict[str, Tensor] = {}  # Store subcomponents in dict for telemetry

    @abstractmethod
    def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Inheriting classes must override this to compute the inference-error component of the loss.
        The target for the predictions should be extracted from the volume; whether this be the voxelwise X0s or the value of the `target` attribute.

        Arguments:
            pred: the predictions from the inference
            inv_pred_weight: weight(s) that should divide the (unreduced) loss between the predictions and targets
            volume: Volume containing the passive volume that was being predicted

        Returns:
            The reduced loss for the performance of the predictions
        """

        pass

    def forward(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Computes the loss for the predictions of a single volume using the current state of the detector

        Arguments:
            pred: the predictions from the inference
            inv_pred_weight: weight(s) that should divide the (unreduced) loss between the predictions and targets
            volume: Volume containing the passive volume that was being predicted and the detector being optimised

        Returns:
            The loss for the predictions and detector
        """

        self.sub_losses = {}
        self.sub_losses["error"] = self._get_inference_loss(pred, inv_pred_weight, volume)
        self.sub_losses["cost"] = self._get_cost_loss(volume)
        return self.sub_losses["error"] + self.sub_losses["cost"]

    def _get_budget_coef(self, cost: Tensor) -> Tensor:
        r"""
        Computes the budget loss term from the current cost of the detectors.
        Switch-on near target budget, plus linear/smooth increase above budget

        Arguments:
            cost: the current cost of the detector in currency units

        Returns:
            The budget loss component
        """

        if self.target_budget is None:
            return cost.new_zeros(1)

        if self.steep_budget:
            d = self.budget_smoothing * (cost - self.target_budget) / self.target_budget
            if d <= 0:
                return 2 * torch.sigmoid(d)
            else:
                return 1 + (d / 2)
        else:
            d = cost - self.target_budget
            return (2 * torch.sigmoid(self.budget_smoothing * d / self.target_budget)) + (F.relu(d) / self.target_budget)

    def _compute_cost_coef(self, inference: Tensor) -> None:
        r"""
        If the cost coefficient is None, will set it equal the current value of the inference-error loss

        Arguments:
            inference: the inference error component of the loss
        """

        self.cost_coef = inference.detach().clone()
        print(f"Automatically setting cost coefficient to {self.cost_coef}")

    def _get_cost_loss(self, volume: Volume) -> Tensor:
        r"""
        Computes the budget term of the loss, dependent on the current cost of the detectors

        Arguments:
            volume: Volume containing the detectors being optimised

        Returns:
            The reduced loss term for the cost of the detectors
        """

        if self.cost_coef is None:
            self._compute_cost_coef(self.sub_losses["error"])
        cost = volume.get_cost()
        cost_loss = self._get_budget_coef(cost) * self.cost_coef
        if self.debug:
            print(
                f'cost {cost}, cost coef {self.cost_coef}, budget coef {self._get_budget_coef(cost)}. error loss {self.sub_losses["error"]}, cost loss {cost_loss}'
            )
        return cost_loss


class VoxelX0Loss(AbsDetectorLoss):
    r"""
    Loss function designed for tasks where the voxelwise X0 value must be predicted as floats.
    Inference-error component of the loss is the squared-error on X0 predictions, averaged over all voxels (MSE)

    The total loss consists of:
        - The MSE
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Arguments:
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Computes the MSE of the predictions against the true voxelwise X0s.

        Arguments:
            pred: (z,x,y) voxelwise X0 predictions from the inference
            inv_pred_weight: weight that divides the unreduced squared error loss between the predictions and targets, prior to averaging
            volume: Volume containing the passive volume that was being predicted

        Returns:
            The MSE for the predictions
        """

        true_x0 = volume.get_rad_cube()
        return torch.mean(F.mse_loss(pred, true_x0, reduction="none") / inv_pred_weight)


class AbsMaterialClassLoss(AbsDetectorLoss):
    r"""
    Abstract base class for cases in which the task is to classify materials in the passive volumes, or some other aspect of the volumes.
    The targets returned by the volume are expected to be float X0s, and are converted to class IDs using an X0 to ID map.

    The loss consists of:
        - A component that quantifies the performance of the predictions made via the detectors
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The performance component (error) should ideally be as close to the final task that the detector will be performing,
    and will depend on the output of the inference algorithm used

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Inheriting classes will need to at least override the `_get_inference_loss` method.

    Arguments:
        x02id: Dictionary mapping float X0 targets to integer class IDs
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def __init__(
        self,
        *,
        x02id: Dict[float, int],
        target_budget: float,
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        super().__init__(target_budget=target_budget, budget_smoothing=budget_smoothing, cost_coef=cost_coef, steep_budget=steep_budget, debug=debug)
        self.x02id = x02id


class VoxelClassLoss(AbsMaterialClassLoss):
    r"""
    Loss function designed for tasks where the voxelwise material class ID must be classified.
    Inference-error component of the loss is the negative log-likelihood on log class-probabilities, averaged over all voxels (NLL)

    Predictions should be provided as log-softmaxed class probabilities per voxel, with shape (1,classes,voxels).
    The ordering of the "flattened" voxels should match that of `volume.get_rad_cube().flatten()`

    The total loss consists of:
        - The NLL
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Arguments:
        x02id: Dictionary mapping float X0 targets to integer class IDs
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Computes the NLL of the log-probabilities against the true voxelwise classes.

        Arguments:
            pred: (1,classes,voxels) log probabilities for voxel class IDs
            inv_pred_weight: weight that divides the unreduced NLL loss between the predictions and targets, prior to averaging
            volume: Volume containing the passive volume that was being predicted

        Returns:
            The mean NLL for the predictions
        """

        true_x0 = volume.get_rad_cube()
        for x0 in true_x0.unique():
            true_x0[true_x0 == x0] = self.x02id[min(self.x02id, key=lambda x: abs(x - x0))]
        true_x0 = true_x0.long().flatten()[None]
        return torch.mean(F.nll_loss(pred, true_x0, reduction="none") / inv_pred_weight)


class VolumeClassLoss(AbsMaterialClassLoss):
    r"""
    Loss function designed for tasks where some overall target of the passive volume must be classified, and the target of the volume is encoded as a float X0.
    E.g. what is the material of a large block in the volume.

    The Inference-error component of the loss depends on shape of predictions provided:
        If the predictions are of shape (1,classes,voxels), they will be interpreted as multi-class log-probabilities and the negative log-likelihood computed
        If the predictions are of shape (1,1,voxels), they will be interpreted as binary class probabilities and the binary cross-entropy computed

    The ordering of the "flattened" voxels should match that of `volume.get_rad_cube().flatten()`

    The total loss consists of:
        - The NLL or BCE
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Arguments:
        x02id: Dictionary mapping float X0 targets to integer class IDs
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Computes the NLL of the log-probabilities against the true voxelwise classes.

        Arguments:
            pred: (1,classes,voxels) log probabilities for voxel class IDs, or (1,1,voxels) probabilities for voxels being of class 1
            inv_pred_weight: weight that divides the unreduced NLL|BCE loss between the predictions and targets, prior to averaging
            volume: Volume containing the passive volume that was being predicted

        Returns:
            The mean NLL|BCE for the predictions
        """

        targ = volume.target.clone()
        for x0 in targ.unique():
            targ[targ == x0] = self.x02id[min(self.x02id, key=lambda x: abs(x - x0))]
        loss = F.nll_loss(pred, targ.long(), reduction="none") if pred.shape[1] > 1 else F.binary_cross_entropy(pred, targ[:, None].float(), reduction="none")
        return torch.mean(loss / inv_pred_weight)


class VolumeIntClassLoss(AbsDetectorLoss):
    r"""
    Loss function designed for tasks where some overall integer target of the passive volume must be classified,
    and the values of this target are quantifiably comparable (i.e. the integers are treatable as numbers not just categorical codes).
    E.g. Predicting how many layers of the passive volume are filled with a given material.

    The Inference-error component of the loss computed as the :meth:`~tomopt.optimisation.loss.sub_losses.integer_class_loss`.
    Predictions should be provided as probabilities for every possible integer target
    The target from the volume can be converted to an integer (e.g. height to layer ID) using a `targ2int` function

    The total loss consists of:
        - The integer class loss (ICL)
        - An optional component that relates to the cost of the detector
    The total loss is the sum of these, with the cost-component being rescaled by a coefficient characterising its relative importance.

    The optional cost component is included as a budget weighting, which gradually increases with the current cost up to a predefined budget,
    after which it increases rapidly, but smoothly.
    Be default, the budget is based on a sigmoid centred at the budget, which linearly increases after the budget is exceeded.
    A less steep version is selectable, which flattens out slightly for high costs.

    Arguments:
        target_budget: If not None, will include a cost component in the loss configured for the specified budget.
            Should be specified in the same currency units as the detector cost.
        budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
        cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
            If set to None, it will be set equal to the inference-error computed the first time the loss is computed
        steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
            otherwise the budget term will flatten off for very high costs
        debug: If True, will print out information about the loss whenever it is evaluated
    """

    def __init__(
        self,
        *,
        targ2int: Callable[[Tensor, Volume], Tensor],
        pred_int_start: int,
        use_mse: bool,
        target_budget: float,
        budget_smoothing: float = 10,
        cost_coef: Optional[Union[Tensor, float]] = None,
        steep_budget: bool = True,
        debug: bool = False,
    ):
        r"""
        Arguments:
            targ2int: function to convert volume targets to integers to classify
            pred_int_start: the integer that the zeroth probability in predictions corresponds to
            use_mse: passed to :meth:`~tomopt.optimisation.loss.sub_losses.integer_class_loss`
            target_budget: If not None, will include a cost component in the loss configured for the specified budget.
                Should be specified in the same currency units as the detector cost.
            budget_smoothing: controls how quickly the budget term rises with cost; lower values => slower rise
            cost_coef: Balancing coefficient used to multiply the budget term prior to its addition to the error component of the loss.
                If set to None, it will be set equal to the inference-error computed the first time the loss is computed
            steep_budget: If True, will use a linearly increasing budget term when the budget is exceeded,
                otherwise the budget term will flatten off for very high costs
            debug: If True, will print out information about the loss whenever it is evaluated
        """
        super().__init__(target_budget=target_budget, budget_smoothing=budget_smoothing, cost_coef=cost_coef, steep_budget=steep_budget, debug=debug)
        self.targ2int, self.pred_int_start, self.use_mse = targ2int, pred_int_start, use_mse

    def _get_inference_loss(self, pred: Tensor, inv_pred_weight: Tensor, volume: Volume) -> Tensor:
        r"""
        Computes the ICL of the integer probabilities against the true target integer.

        Arguments:
            pred: (1,*,integers) integer probabilities
            inv_pred_weight: weight that divides the unreduced ICL between the predictions and targets, prior to averaging
            volume: Volume containing the passive volume that was being predicted

        Returns:
            The mean ICL for the predictions
        """

        int_targ = self.targ2int(volume.target.clone(), volume)
        loss = integer_class_loss(pred, int_targ, pred_start_int=self.pred_int_start, use_mse=self.use_mse, reduction="none")
        return torch.mean(loss / inv_pred_weight)
