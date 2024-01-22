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

def compute_bins_edges(bins_count:int, range_x, range_y):
    step_x = (range_x[1]- range_x[0])/bins_count
    step_y = (range_y[1]- range_y[0])/bins_count
    bins_edges_x = np.arange(range_x[0], range_x[1]+step_x, step_x)
    bins_edges_y = np.arange(range_y[0], range_y[1]+step_y, step_y)
    return np.asarray([bins_edges_x,bins_edges_y], dtype=np.float64)