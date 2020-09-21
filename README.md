# numba_alloc
A numba njit heuristic for a variant of the 
[integer programming problem](https://en.wikipedia.org/wiki/Integer_programming).

We we want to find the <img src="https://latex.codecogs.com/gif.latex?M_{ij}" /> values for 2D binary 
matrix *M* such that the element-wise multiplication of *M* with the weight matrix *W* of the same shape
is maximized.  
This is done under the constraints that 
* the sum of the rows of *M* cannot exceed the values given in the vector *R*
* the sum of the columns of *M* cannot exceed the values given in the vector *C*

Also, we assume that the number of rows in the matrix is far larger than the number of columns. 

Basicaly:

* Maximize  
<img src="https://latex.codecogs.com/gif.latex?\sum_{i, j} m_{i, j} * W_{i, j}" />
* with   
<img src="https://latex.codecogs.com/gif.latex?\sum_{j} m_{i, j} \le R_{i} \forall i \in I" />
* and   
<img src="https://latex.codecogs.com/gif.latex?\sum_{i} m_{i, j} \le C_{j} \forall j \in J" />
* and   
<img src="https://latex.codecogs.com/gif.latex?I >> J" />


