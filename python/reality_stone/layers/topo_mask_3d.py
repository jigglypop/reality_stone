from typing import Dict, List, Tuple

import torch


def _to_tensor(x, dtype=torch.float32, device=None):
    t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def quantize_rows_cols(
    bboxes: List[List[int]],
    image_size: Tuple[int, int],
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map FUNSD-style relative bboxes ([0,1000]) to coarse row/col grid indices.

    Args:
        bboxes: list of [x0,y0,x1,y1] in [0,1000]
        image_size: (W,H)
        grid_rows: number of row bins
        grid_cols: number of col bins

    Returns:
        row_ids: (T,) long tensor
        col_ids: (T,) long tensor
    """
    W, H = image_size
    arr = _to_tensor(bboxes, dtype=torch.float32)
    # normalize to [0,1]
    x0 = arr[:, 0] / 1000.0
    y0 = arr[:, 1] / 1000.0
    x1 = arr[:, 2] / 1000.0
    y1 = arr[:, 3] / 1000.0
    # use box centers
    xc = (x0 + x1) * 0.5
    yc = (y0 + y1) * 0.5
    row_ids = torch.clamp((yc * grid_rows).long(), 0, grid_rows - 1)
    col_ids = torch.clamp((xc * grid_cols).long(), 0, grid_cols - 1)
    return row_ids, col_ids


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (T,2), b: (T,2)
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)


def build_relations_from_bboxes(
    bboxes: List[List[int]],
    image_size: Tuple[int, int],
    k_row: int = 16,
    k_col: int = 8,
    k_cell: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Build simple row/col/cell neighbor indices for one page.

    Returns mapping rel -> idx (1,T,k)
    """
    T = len(bboxes)
    if T == 0:
        raise ValueError("empty bboxes")
    W, H = image_size
    arr = _to_tensor(bboxes, dtype=torch.float32)
    # centers in pixel space
    xc = ((arr[:, 0] + arr[:, 2]) * 0.5) * (W / 1000.0)
    yc = ((arr[:, 1] + arr[:, 3]) * 0.5) * (H / 1000.0)
    coords = torch.stack([xc, yc], dim=-1)  # (T,2)

    # row neighbors: sort by |y diff| then |x diff|
    ydiff = (yc[:, None] - yc[None, :]).abs()
    xdiff = (xc[:, None] - xc[None, :]).abs()
    score_row = ydiff * 1.0 + xdiff * 0.01
    idx_row = torch.argsort(score_row, dim=-1)[:, : max(1, k_row)]

    # col neighbors: sort by |x diff| then |y diff|
    score_col = xdiff * 1.0 + ydiff * 0.01
    idx_col = torch.argsort(score_col, dim=-1)[:, : max(1, k_col)]

    # cell neighbors: nearest euclidean
    d2 = _pairwise_l2(coords, coords)
    idx_cell = torch.argsort(d2, dim=-1)[:, : max(1, k_cell)]

    # Expand to batch=1
    return {
        "row": idx_row.unsqueeze(0).to(torch.long),
        "col": idx_col.unsqueeze(0).to(torch.long),
        "cell": idx_cell.unsqueeze(0).to(torch.long),
    }


__all__ = [
    "quantize_rows_cols",
    "build_relations_from_bboxes",
]


