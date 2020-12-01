# gpuschool2018

Examples and teaching material for the July 2018 "Introduction to GPU programming" summer school at the University of Warwick. 

To run the notebooks you will need a CUDA capable GPU card. The examples use double precision arithmetic and assume a GPU with a compute capability of 2.0 or higher. The school was conducted using Kepler K20c and K80 hardware. 

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with version 9.1)
* [Python 3](https://www.python.org/downloads/)
* [Numpy](http://www.numpy.org/)
* [Numba](https://numba.pydata.org/) 
* [pyculib](https://github.com/numba/pyculib) 
* [Pillow](https://python-pillow.org/) 
* [Matplotlib](https://matplotlib.org/)
* [Jupyter](http://jupyter.org/)

This software is available on all Linux hosts managed by the [Scientific Computing Research Technology Plaform](https://warwick.ac.uk/research/rtp/sc) at the [University of Warwick](https://warwick.ac.uk/) via environment modules,
```
$ module load GCC/6.4.0-2.28 OpenMPI Python OpenBLAS CUDA numba pyculib Pillow IPython matplotlib
```
or (2019 modules)
```
$ module load GCC/7.3.0-2.30 OpenMPI Python OpenBLAS CUDA numba pyculib Pillow IPython matplotlib
```
or (2020 modules)
```
$ module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 Python OpenBLAS numba pyculib Pillow IPython matplotlib
```

Porgs are cool.
