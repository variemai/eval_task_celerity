
# Table of Contents

1.  [Celerity Evaluation Project:](#org3577e51)
2.  [Build](#orgb39233d)
3.  [Run](#org194d9c4)



<a id="org3577e51"></a>

# Celerity Evaluation Project

For the purposes of this project I attempted to implement two kernels using the Celerity runtime and the hipSYCL implementation of SYCL. The two kernels are **LU decomposition** and **Matrix Multiplication**. I decided to include both kernels as the implementation of **LU decomposition** kernel was not succesful. Please, refer to [Report](./Report.pdf) for the report of my experience using the Celerity runtime.


<a id="orgb39233d"></a>

# Build

The source code for lu and matrix multplication kernels is provided in lu.cc and mmult.cc files. A CMakeLists.txt file is provided for building both of the kernels with cmake.
This project is build and run in Linux.


<a id="org194d9c4"></a>

# Run

-   LU is run simply by running the executable `./lu`. If you provide it with an extra arg `./lu <int>` the dimensions of the matrices will be modified to the int argument's value. The default value is 4.
-   Matrix Multiplication is run in the same fashion as LU, `./mmult` and `./mmult <int>`, the default value is 2048 . It can also be run with mpi in the case of a multinode compute cluster `mpirun -np <nnodes> ./mmult <int>`
-   Both programs use square matrices
-   Take note that there is no error check in the provided argument

