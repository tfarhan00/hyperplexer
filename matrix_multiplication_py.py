from typing import List

def matrix_multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    """
    Multiply two matrices.

    Parameters:
        mat1 (List[List[int]]): The first matrix.
        mat2 (List[List[int]]): The second matrix.

    Returns:
        List[List[int]]: The result of the matrix multiplication.
    """
    n = len(mat1)
    m = len(mat2)
    p = len(mat2[0])
    result = [[0] * p for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += mat1[i][k] * mat2[k][j]
    
    return result
