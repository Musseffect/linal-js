import Matrix from "../../dense/denseMatrix";
import { assert } from "../../utils";

export interface givensCoeffs {
    c: number, s: number, r: number
};

// compute coefficients for givens matrix
export function givens(x: number, y: number): givensCoeffs {
    let c = 0.0;
    let s = 0.0;
    let r = 0.0;
    if (y == 0) {
        c = Math.sign(x);
        r = Math.abs(x);
    } else if (x == 0) {
        s = -Math.sign(y);
        r = Math.abs(y);
    } else if (Math.abs(x) > Math.abs(y)) {
        let t = y / x;
        let u = Math.sign(x) * Math.sqrt(1 + t * t);
        c = 1.0 / u;
        s = -c * t;
        r = x * u;
    } else {
        let t = x / y;
        let u = Math.sign(y) * Math.sqrt(1 + t * t);
        s = -1.0 / u;
        c = t / u;
        r = y * u;
    }
    return { c, s, r };
}
/** Multiply A by rotation matrix
 *        j ... i
 *  j |   c    -s
 *  . |
 *  i |   s     c
*/
export function applyGivensFromLeft(A: Matrix, givens: givensCoeffs, i: number, j: number): void {
    assert(i > j, "Incorrect order of indices");
    assert(A.numRows() > i, "Incorrect number of rows");
    for (let col = 0; col < A.numCols(); ++col) {
        let a = A.get(j, col);
        let b = A.get(i, col);
        A.set(j, col, givens.c * a - givens.s * b);
        A.set(i, col, givens.s * a + givens.c * b);
    }
}
/** Multiply A by transposed rotation matrix
 *        j ... i
*  j |    c     s
*  . |
*  i |   -s     c
*/
export function applyTransposeGivensFromLeft(A: Matrix, givens: givensCoeffs, i: number, j: number): void {
    assert(i > j, "Incorrect order of indices");
    assert(A.numRows() > i, "Incorrect number of rows");
    for (let col = 0; col < A.numCols(); ++col) {
        let a = A.get(j, col);
        let b = A.get(i, col);
        A.set(j, col, givens.c * a + givens.s * b);
        A.set(i, col, -givens.s * a + givens.c * b);
    }
}
/** Multiply A by transposed rotation matrix
 *        j ... i
*  j |    c     s
*  . |
*  i |   -s     c
*/
export function applyTransposeGivensFromRight(A: Matrix, givens: givensCoeffs, i: number, j: number) {
    assert(i > j, "Incorrect order of indices");
    assert(A.numCols() > i, "Incorrect number of cols");
    for (let row = 0; row < A.numRows(); ++row) {
        let a = A.get(row, j);
        let b = A.get(row, i);
        A.set(row, j, givens.c * a - givens.s * b);
        A.set(row, i, givens.s * a + givens.c * b);
    }
}
/** Multiply A by rotation matrix
 *        j ... i
 *  j |   c    -s
 *  . |
 *  i |   s     c
*/
export function applyGivensFromRight(A: Matrix, givens: givensCoeffs, i: number, j: number) {
    assert(i > j, "Incorrect order of indices");
    assert(A.numCols() > i, "Incorrect number of cols");
    for (let row = 0; row < A.numRows(); ++row) {
        let a = A.get(row, j);
        let b = A.get(row, i);
        A.set(row, j, givens.c * a + givens.s * b);
        A.set(row, i, -givens.s * a + givens.c * b);
    }
}

/**
 *  Make matrix
 * 
 * [c, -s]\
 * [s, c]
 *  */

export function makeGivensMatrix(givens: givensCoeffs, matrixSize: number, i: number, j: number): Matrix {
    assert(i > j, "Incorrect order of indices");
    assert(matrixSize > i, "Incorrect matrix size");
    let m = Matrix.empty(matrixSize, matrixSize);
    m.set(j, j, givens.c);
    m.set(j, i, -givens.s);
    m.set(i, i, givens.c);
    m.set(i, j, givens.s);
    for (let row = 0; row < matrixSize; ++row) {
        if (row != i && row != j)
            m.set(row, row, 1);
    }
    return m;
}