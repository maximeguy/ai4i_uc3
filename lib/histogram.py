import numpy as np

def compute_bins_count(method:str, n:int):
    """Compute the number of bins from the selected method for n elements

    Args:
        method (str): _description_
        n (int): _description_

    Returns:
        _type_: _description_
    """
    if (method == "sturges"):
        bins_count = 1 + np.log2(n)
    elif(method == 'rice'):
        bins_count = 2*np.cbrt(n)
    elif (method == "sqrt"):
        bins_count = np.sqrt(n)
    
    print(f"Computed bins_count with {method} method for {n} elements): ",int(bins_count))
    return int(bins_count)

def compute_bins_edges(bins_count: int, components_ranges):
    """Compute the number of bins from the selected method for n elements

    Args:
        bins_count (int): Number of histogram bins
        component_range (np.ndarray): Value range of a component

    Returns:
        np.ndarray: Array of edges of each histogram bin
    """
    if len(components_ranges.shape) == 1:
        step = (components_ranges[1] - components_ranges[0]) / (bins_count)
        bins_edges = np.arange(components_ranges[0], components_ranges[1] + step, step)
    else:
        bins_edges = np.zeros((len(components_ranges), bins_count + 1), dtype=int)
        for i, cr in enumerate(components_ranges):
            step = (cr[1] - cr[0]) / (bins_count)
            bins_edges[i] = np.arange(cr[0], cr[1] + step, step)
    return bins_edges