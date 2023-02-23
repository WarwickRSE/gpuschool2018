# gpuschool2018

Examples and teaching material for the July 2018 "Introduction to GPU programming" summer school at the University of Warwick. 

To run the notebooks you will need a CUDA capable GPU card. The examples use double precision arithmetic and assume a GPU with a compute capability of 2.0 or higher. The school was conducted using Kepler K20c and K80 hardware. 

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with version 9.1-11.1)
* [Python 3](https://www.python.org/downloads/)
* [Numpy](http://www.numpy.org/)
* [Numba](https://numba.pydata.org/) 
* [CuPy](https://cupy.dev) 
* [Pillow](https://python-pillow.org/) 
* [Matplotlib](https://matplotlib.org/)
* [Jupyter](http://jupyter.org/)

This software is available on all Linux hosts managed by the [Scientific Computing Research Technology Plaform](https://warwick.ac.uk/research/rtp/sc) at the [University of Warwick](https://warwick.ac.uk/) via environment modules,

On the (old) CentOS 7 desktop use:
```
$ module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 Python OpenBLAS numba CuPy Pillow IPython matplotlib
```
One new (new) Rocky 9 desktop use:
```
$ module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0 Python numba CuPy Pillow IPython matplotlib
```


Porgs are cool.
