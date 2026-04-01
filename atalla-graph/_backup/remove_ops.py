import torch
import torch.fx
import torch.nn as nn

#these transforms will operate on GraphModule instead of modules
def remove_ops(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    
    # Modify gm.graph

    # <...>

    gm.graph.lint() #check if well-formed
    gm.recompile()

    return gm