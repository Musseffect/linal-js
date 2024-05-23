import Matrix from "../../dense/denseMatrix";
import { assert, sign } from "../../utils";
import Vector from "../../dense/vector";

function processHouseholderVectorInplace(v: Vector, pivotIdx: number = 0): Vector {
    let ro = -sign(v.get(pivotIdx));
    let xNormSqr = v.squaredLength();
    let xNorm = Math.sqrt(xNormSqr);
    let firstElement = v.get(pivotIdx);
    v.set(pivotIdx, v.get(pivotIdx) - ro * xNorm);
    v.scaleSelf(1.0 / Math.sqrt(xNormSqr - firstElement * firstElement + v.get(pivotIdx) * v.get(pivotIdx)));
    return v;
}

export function makeHouseholderMatrix(v: Vector, startIdx: number, totalSize: number) {
    assert(totalSize >= startIdx + v.size(), "Incorrect sizes");
    let matrix = Matrix.identity(totalSize);
    for (let i = 0; i < v.size(); ++i) {
        for (let j = i; j < v.size(); ++j) {
            let value = matrix.get(i + startIdx, j + startIdx) - 2 * v.get(i) * v.get(j);
            matrix.set(i + startIdx, j + startIdx, value);
            if (i != j)
                matrix.set(j + startIdx, i + startIdx, value);
        }
    }
    return matrix;
}

export function calcHouseholderVectorCol(A: Matrix, row: number, col: number, size?: number, pivotIdx: number = 0): Vector {
    assert(row < A.numRows(), "Incorrect row");
    if (size == undefined)
        size = A.numRows() - row;
    else
        assert(size <= A.numRows() - row, "Incorrect size");
    let v = A.subColumn(row, col, size);
    return processHouseholderVectorInplace(v, pivotIdx);
}

export function calcHouseholderVectorRow(A: Matrix, row: number, col: number, size?: number, pivotIdx: number = 0): Vector {
    assert(col < A.numCols(), "Incorrect row");
    if (size == undefined)
        size = A.numCols() - col;
    else
        assert(size <= A.numCols() - col, "Incorrect size");
    let v = A.subRow(row, col, size);
    return processHouseholderVectorInplace(v, pivotIdx);
}

export function applyHouseholderFromLeft(v: Vector, A: Matrix, idx: number) {
    for (let col = 0; col < A.numCols(); ++col) {
        let vDotX = 0.0;
        for (let row = idx; row < idx + v.size(); ++row)
            vDotX += v.get(row - idx) * A.get(row, col);
        vDotX *= 2;
        for (let row = idx; row < idx + v.size(); ++row)
            A.set(row, col, A.get(row, col) - v.get(row - idx) * vDotX);
    }
}

export function applyHouseholderFromRight(v: Vector, A: Matrix, idx: number) {
    for (let row = 0; row < A.numRows(); ++row) {
        let vDotX = 0.0;
        for (let col = idx; col < idx + v.size(); ++col)
            vDotX += v.get(col - idx) * A.get(row, col);
        vDotX *= 2;
        for (let col = idx; col < idx + v.size(); ++col)
            A.set(row, col, A.get(row, col) - v.get(col - idx) * vDotX);
    }
}