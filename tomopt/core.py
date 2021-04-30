import torch

__all__ = ["DEVICE", "SCATTER_COEF_A", "SCATTER_COEF_B", "X0"]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

SCATTER_COEF_A = 0.0136

SCATTER_COEF_B = 0.038

X0 = {
    "beryllium": 0.3528,  # m
    "carbon": 0.1932,
    "aluminium": 0.08897,
    "silicon": 0.0937,
    "iron": 0.01757,
    "copper": 0.01436,
    "lead": 0.005612,
    # 'air':312.22
}
