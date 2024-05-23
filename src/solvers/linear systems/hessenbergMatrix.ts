import Matrix from "../../dense/denseMatrix";
import { assert, sign } from "../../utils";
import Vector from "../../dense/vector";
import { applyHouseholderFromLeft } from "./hausholderReflection";

// should be 4/3N^3 + O(N^2)
export function makeTridiagonalInplace(A: Matrix, Q?: Matrix): Matrix {
    if (Q) {
        Q.setFromMatrix(Matrix.identity(A.numRows()));
    }
    for (let outerCol = 0; outerCol + 2 < A.numCols(); ++outerCol) {
        let shift = outerCol + 1;
        let u = A.subRow(outerCol, shift, A.numRows() - shift);
        let xNormSqr = u.squaredLength();
        let xNorm = Math.sqrt(xNormSqr);
        let firstElement = u.get(0);
        let ro = -sign(firstElement);
        // first element of the column is alpha, elements below are zero 
        let alpha = ro * xNorm;
        if (alpha == 0) continue;
        u.set(0, firstElement - alpha);
        u.scaleSelf(1.0 / Math.sqrt(xNormSqr - firstElement * firstElement + u.get(0) * u.get(0)));
        // premultiply all rows
        // first (col + 1) rows won't change
        // set first column
        A.set(shift, outerCol, alpha);
        A.set(outerCol, shift, alpha);
        for (let row = shift + 1; row < A.numRows(); ++row) {
            A.set(row, outerCol, 0);
            A.set(outerCol, row, 0);
        }
        // calc pre and post multiplicaiton simultaniously
        let v = Vector.empty(A.numRows() - shift);
        // calc 2 * A * u and store in v
        for (let row = shift; row < A.numRows(); ++row) {
            let value = 0.0;
            for (let col = shift; col < A.numCols(); ++col) {
                value += A.get(row, col) * u.get(col - shift);
            }
            v.set(row - shift, 2 * value);
        }
        let utAu = Vector.dot(u, v);
        // calc v = 2 * A * u - 2u * (ut * A * u)
        for (let idx = 0; idx < v.size(); ++idx) {
            v.set(idx, v.get(idx) - u.get(idx) * utAu);
        }

        // calc A - uvT - vutv
        for (let row = shift; row < A.numRows(); ++row) {
            for (let col = row; col < A.numCols(); ++col) {
                A.set(row, col, A.get(row, col) - v.get(row - shift) * u.get(col - shift) - v.get(col - shift) * u.get(row - shift));
            }
        }
        // preserve symmetry
        for (let row = shift; row < A.numRows(); ++row) {
            for (let col = row + 1; col < A.numCols(); ++col)
                A.set(col, row, A.get(row, col));
        }
        if (Q) {
            applyHouseholderFromLeft(u, Q, shift);
        }
    }
    return A;
}

export function makeTridiagonalInplaceAlt(A: Matrix, Q?: Matrix): Matrix {
    assert(A.isSymmetric(), "A must be symmetric");
    for (let outerCol = 0; outerCol + 2 < A.numCols(); ++outerCol) {
        let shift = outerCol + 1;
        let xNormSqr = 0;
        for (let col = shift; col < A.numCols(); ++col)
            xNormSqr += Math.pow(A.get(outerCol, col), 2);
        let xNorm = Math.sqrt(xNormSqr);
        let firstElement = A.get(outerCol, shift);
        let ro = -sign(firstElement);
        // first element of the column is alpha, elements below are zero 
        let alpha = ro * xNorm;
        if (alpha == 0) continue;
        let newFirstElement = firstElement - alpha;
        A.set(outerCol, shift, firstElement - alpha);
        let newLength = Math.sqrt(xNormSqr - firstElement * firstElement + newFirstElement * newFirstElement);
        for (let col = shift; col < A.numCols(); ++col)
            A.set(outerCol, col, A.get(outerCol, col) / newLength);
        //let u = A.subRow(outerCol, shift, A.numRows() - shift);
        // premultiply all rows
        // first (col + 1) rows won't change
        // set first column
        A.set(shift, outerCol, alpha);
        for (let row = shift + 1; row < A.numRows(); ++row)
            A.set(row, outerCol, 0);
        // calc pre and post multiplicaiton simultaniously
        let v = Vector.empty(A.numRows() - shift);
        // calc 2 * A * u and store in v
        for (let row = shift; row < A.numRows(); ++row) {
            let value = 0.0;
            for (let col = shift; col < A.numCols(); ++col) {
                value += A.get(row, col) * /*u*/A.get(outerCol, col);
            }
            v.set(row - shift, 2 * value);
        }
        let utAu = 0;
        for (let col = shift; col < A.numCols(); ++col)
            utAu += /*u*/A.get(outerCol, col) * v.get(col - shift);
        // calc v = 2 * A * u - 2u * (ut * A * u)
        for (let idx = 0; idx < v.size(); ++idx) {
            v.set(idx, v.get(idx) - /*u*/A.get(outerCol, idx + shift) * utAu);
        }

        // calc A - uvT - vutv
        for (let row = shift; row < A.numRows(); ++row) {
            for (let col = row; col < A.numCols(); ++col) {
                A.set(row, col, A.get(row, col) - v.get(row - shift) * /*u*/A.get(outerCol, col) - v.get(col - shift) * /*u*/A.get(outerCol, row));
            }
        }
        // preserve symmetry
        for (let row = shift; row < A.numRows(); ++row) {
            for (let col = row + 1; col < A.numCols(); ++col)
                A.set(col, row, A.get(row, col));
        }
    }
    if (Q) {
        //Q.setFromMatrix(Matrix.identity(Q.numRows()));
        // Set initial Q with values for 2x2 Q_k
        let v1 = A.get(A.numRows() - 3, A.numCols() - 2);
        let v2 = A.get(A.numRows() - 3, A.numCols() - 1);
        let v22 = v2 * v2;
        let v11 = v1 * v1;
        let v12 = v1 * v2;

        //Q.set(A.numRows() - 2, A.numCols() - 3, 0);
        Q.set(A.numRows() - 2, A.numCols() - 2, 1 - 2 * v11);
        Q.set(A.numRows() - 2, A.numCols() - 1, - 2 * v12);

        //Q.set(A.numRows() - 1, A.numCols() - 3, 0);
        Q.set(A.numRows() - 1, A.numCols() - 2, - 2 * v12);
        Q.set(A.numRows() - 1, A.numCols() - 1, 1 - 2 * v22);

        Q.set(A.numRows() - 3, A.numCols() - 3, 1);
        //Q.set(A.numRows() - 3, A.numCols() - 2, 0);
        //Q.set(A.numRows() - 3, A.numCols() - 1, 0);

        // accumulate v in A and construct Q at the end by multiplying from last to first
        for (let aRow = A.numCols() - 4; aRow >= 0; --aRow) {
            let qCol = aRow + 1;
            Q.set(aRow, aRow, 1);
            let v1_2 = 2 * A.get(aRow, qCol);
            // set the first row of non-identity part of Q
            Q.set(qCol, qCol, Q.get(qCol, qCol) - v1_2 * A.get(aRow, qCol));
            for (let col = qCol + 1; col < A.numCols(); ++col) {
                //Q.set(qCol, col, Q.get(qCol, col) - v1_2 * A.get(aRow, col));
                Q.set(qCol, col, - v1_2 * A.get(aRow, col));
            }
            for (let row = qCol + 1; row < A.numRows(); ++row) {
                let vDotX = 0.0;
                for (let col = qCol + 1; col < A.numCols(); ++col) {
                    vDotX += A.get(aRow, col) * Q.get(row, col);
                }
                vDotX *= 2;
                Q.set(row, qCol, - A.get(aRow, qCol) * vDotX);
                for (let col = qCol + 1; col < A.numCols(); ++col) {
                    Q.set(row, col, Q.get(row, col) - A.get(aRow, col) * vDotX);
                }
            }
        }
        for (let i = 1; i < A.numRows(); ++i) {
            Q.set(i, 0, 0);
            Q.set(0, i, 0);
        }
    }
    // set correct values for upper triangle of A
    for (let row = 0; row + 2 < A.numRows(); ++row)
        for (let col = row + 1; col < A.numCols(); ++col)
            A.set(row, col, A.get(col, row));
    return A;
}

export function makeTridiagonal(A: Matrix, Q?: Matrix) {
    return makeTridiagonalInplace(A.clone(), Q);
}

export function makeTridiagonalAlt(A: Matrix, Q?: Matrix) {
    return makeTridiagonalInplaceAlt(A.clone(), Q);
}

function makeHessenbergInplaceWithQ(A: Matrix, Q: Matrix): Matrix {
    Q.setFromMatrix(Matrix.identity(A.numRows()));
    if (A.numCols() < 3) return A;
    for (let outerCol = 0; outerCol + 2 < A.numCols(); ++outerCol) {
        let shift = outerCol + 1;
        for (let row = shift; row < A.numRows(); ++row)
            Q.set(shift, row, A.get(row, outerCol));
        let xNormSqr = 0.0;
        for (let row = shift; row < A.numRows(); ++row)
            xNormSqr += Math.pow(A.get(row, outerCol), 2);
        let xNorm = Math.sqrt(xNormSqr);
        let firstElement = Q.get(shift, shift);
        let ro = -sign(firstElement);
        // first element of the column is alpha, elements below are zero 
        let alpha = ro * xNorm;
        if (alpha == 0) continue;
        let newFirstElement = firstElement - alpha;
        Q.set(shift, shift, newFirstElement);
        const factor = 1.0 / Math.sqrt(xNormSqr - firstElement * firstElement + newFirstElement * newFirstElement);
        for (let row = shift; row < A.numRows(); ++row)
            Q.set(shift, row, /*u*/Q.get(shift, row) * factor);
        // premultiply all rows
        // first (col + 1) rows won't change
        // set first column
        A.set(shift, outerCol, alpha);
        for (let row = shift + 1; row < A.numRows(); ++row)
            A.set(row, outerCol, 0);
        // set other columns ~O(2N^2)
        for (let col = shift; col < A.numCols(); ++col) {
            let vDotX = 0.0;
            for (let row = shift; row < A.numRows(); ++row) {
                vDotX += /*u*/Q.get(shift, row) * A.get(row, col);
            }
            vDotX *= 2;
            for (let row = shift; row < A.numRows(); ++row) {
                A.set(row, col, A.get(row, col) - /*u*/Q.get(shift, row) * vDotX);
            }
        }
        for (let row = 0; row < A.numRows(); ++row) {
            let vDotX = 0.0;
            for (let col = shift; col < A.numCols(); ++col) {
                vDotX += /*u*/Q.get(shift, col) * A.get(row, col);
            }
            vDotX *= 2;
            for (let col = shift; col < A.numCols(); ++col) {
                A.set(row, col, A.get(row, col) - /*u*/Q.get(shift, col) * vDotX);
            }
        }
    }
    // set first 2x2 non identity part of matrix Q
    let v1 = Q.get(Q.numRows() - 2, Q.numCols() - 2);
    let v2 = Q.get(Q.numRows() - 2, Q.numCols() - 1);
    let v22 = v2 * v2;
    let v11 = v1 * v1;
    let v12 = v1 * v2;
    Q.set(Q.numRows() - 2, Q.numCols() - 2, 1 - 2 * v11);
    Q.set(Q.numRows() - 1, Q.numCols() - 2, - 2 * v12);
    Q.set(Q.numRows() - 2, Q.numCols() - 1, - 2 * v12);
    Q.set(Q.numRows() - 1, Q.numCols() - 1, 1 - 2 * v22);
    // accumulate v in A and construct Q at the end by multiplying from last to first
    for (let aCol = Q.numCols() - 4; aCol >= 0; --aCol) {
        let qCol = aCol + 1;
        let v1 = Q.get(qCol, qCol);
        let v1_2 = 2 * v1;
        for (let row = qCol + 1; row < Q.numRows(); ++row) {
            let vDotX = 0.0;
            for (let col = qCol + 1; col < Q.numCols(); ++col) {
                vDotX += /*u*/Q.get(qCol, col) * Q.get(row, col);
            }
            vDotX *= 2;
            for (let col = qCol; col < Q.numCols(); ++col) {
                Q.set(row, col, Q.get(row, col) - /*u*/Q.get(qCol, col) * vDotX);
            }
        }
        Q.set(qCol, qCol, 1 - v1 * v1_2);
        // set the first row of non-identity part of Q
        for (let col = qCol + 1; col < Q.numCols(); ++col) {
            Q.set(qCol, col, - v1_2 * /*u*/Q.get(qCol, col));
        }
    }
    return A;
}

// todo: test with nonempty Q, there is a bug in implementation
export function makeHessenbergInplace(A: Matrix, Q?: Matrix): Matrix {
    //if (A.isSymmetric()) return makeTridiagonalInplace(A, Q);
    assert(A.isSquare(), "Non-square matrix");
    // todo: store n-1 elements of v in H and reconstruct Q afterwards: |v| = 1 by applying householder from the right
    // Q = P1*P2...PN-2
    if (Q) return makeHessenbergInplaceWithQ(A, Q);
    for (let outerCol = 0; outerCol + 2 < A.numCols(); ++outerCol) {
        let shift = outerCol + 1;
        let v = A.subColumn(shift, outerCol, A.numRows() - shift);
        let xNormSqr = v.squaredLength();
        let xNorm = Math.sqrt(xNormSqr);
        let firstElement = v.get(0);
        let ro = -sign(firstElement);
        // first element of the column is alpha, elements below are zero 
        let alpha = ro * xNorm;
        if (alpha == 0) continue;
        v.set(0, firstElement - alpha);
        v.scaleSelf(1.0 / Math.sqrt(xNormSqr - firstElement * firstElement + v.get(0) * v.get(0)));
        // premultiply all rows
        // first (col + 1) rows won't change
        // set first column
        A.set(shift, outerCol, alpha);
        // todo (NI): rearrange this to do that in rows
        for (let row = shift + 1; row < A.numRows(); ++row)
            A.set(row, outerCol, 0);
        // set other columns ~O(2N^2)
        for (let col = shift; col < A.numCols(); ++col) {
            let vDotX = 0.0;
            for (let row = shift; row < A.numRows(); ++row) {
                vDotX += v.get(row - shift) * A.get(row, col);
            }
            vDotX *= 2;
            for (let row = shift; row < A.numRows(); ++row) {
                A.set(row, col, A.get(row, col) - v.get(row - shift) * vDotX);
            }
        }
        for (let row = 0; row < A.numRows(); ++row) {
            let vDotX = 0.0;
            for (let col = shift; col < A.numCols(); ++col) {
                vDotX += v.get(col - shift) * A.get(row, col);
            }
            vDotX *= 2;
            for (let col = shift; col < A.numCols(); ++col)
                A.set(row, col, A.get(row, col) - v.get(col - shift) * vDotX);
        }
    }
    return A;
}

/**
 * Compute QHQT = A
 */
export function makeHessenberg(A: Matrix, Q?: Matrix): Matrix {
    return makeHessenbergInplace(A.clone(), Q);
}