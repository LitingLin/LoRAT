def numpy_no_multithreading():
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass
    import os
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    np.seterr(all='raise')
    import warnings
    warnings.filterwarnings('error', "invalid value encountered in cast")
    warnings.filterwarnings('error', "invalid value encountered in true_divide")
