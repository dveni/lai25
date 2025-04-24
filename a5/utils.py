import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

def build_device_mesh(rank: int, world_size: int, pp: int, dp: int = 1, tp: int = 1) -> DeviceMesh:
    pp, dp, tp, world_size = pp, dp, tp, world_size
    
    assert pp >= 1 and tp >= 1 and (dp >= 1 or dp == -1)
    if dp == -1:
        dp = world_size // (pp * tp)
    assert pp * dp * tp == world_size

    dims = [dp, tp]
    names = ["dp", "tp"]

    if pp > 1:
        dims.insert(0, pp)
        names.insert(0, "pp")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if not rank:
        print(f"Building {len(dims)}-D device mesh with {names}, {dims}")
    mesh = init_device_mesh(device_type, dims, mesh_dim_names=tuple(names))
    return mesh