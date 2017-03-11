Date: 2017-03-11  
Author: jkang  
Topic: tensors, ranks and shapes  
Ref: [TensorFlow tutorial](https://www.tensorflow.org/programmers_guide/dims_types)

---

T = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  

T is a **Tensor**.  
T is a **2**-dimensional **Tensor**.  
>Why 2-dimensional? Simply count the number of brackets from left or right. There are two '[' (or two ']') for T.   
>Think about it as a depth (=dimension) of a Tensor.

**n-dimension** is a basic unit for **Tensor**.  
**n-dimension** is also called a **rank 'n'**.  
So, **2-dimensional Tensor** T is also called a **rank 2 Tensor (or 2-Tensor)**.  
**Dimension** and **Rank** are the same term.  

>1-Tensor: e.g. vector, [1,2,3]  
>2-Tensor: e.g. matrix, [[1,2],[3,4]]  
>3-Tensor: e.g. cube, [[[1,2],[3,4]],[[5,6],[7,8]]]    
>n-Tensor: ...  

T has a **shape**=(2, 3).  
>Why (2, 3)? Count number of elements in the smallest bracket and so one, and write the numbers from right to left. That's it!

For example,  
A = [[1,2,3],[4,5,6]]  
B = [[[1,2],[4,5]],[[8,7],[9,1]],[[0,8],[1,3]]]  
C = [[[[4]]],[[[7]]]]  
  
A: 2-dimensional (rank 2) Tensor, shape=(2,3)  
B: 3-dimensional (rank 3) Tensor, shape=(3,2,2)  
C: 4-dimensional (rank 4) Tensor, shape=(2,1,1,1)  




