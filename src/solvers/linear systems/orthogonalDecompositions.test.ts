import Matrix from '../../dense/denseMatrix'
import { SmallTolerance, StopWatch, Tolerance, sign } from '../../utils';
import { givens, applyGivensFromLeft, applyTransposeGivensFromRight, makeGivensMatrix, applyGivensFromRight, applyTransposeGivensFromLeft } from './givensRotation';
import { applyHouseholderFromLeft, applyHouseholderFromRight, calcHouseholderVectorCol, calcHouseholderVectorRow, makeHouseholderMatrix } from './hausholderReflection';
import { makeHessenberg, makeTridiagonal, makeTridiagonalAlt } from './hessenbergMatrix';

import { MatrixGenerator } from '../../dense/matrixGenerator';

import fs from 'fs';
//import * as v8Profiler from 'v8-profiler-next';
import JSGenerator from '../../random/js';
import { jacobiRotation } from './jacobiRotation';
import { OrthogonalDecomposition, OrthogonalDecompositionType, ZeroingMethod } from './qr';

describe('Rotations', () => {
    test('Jacobi', () => {
        let a1 = 2;
        let a2 = 1;
        let b = 11;
        const { c, s } = jacobiRotation(a1, a2, b);
        expect(c * c + s * s).toBeCloseTo(1);
        expect(b * (c * c - s * s) + (a1 - a2) * c * s).toBeCloseTo(0);
        let J = new Matrix([c, s, -s, c], 2, 2);
        let A = new Matrix([a1, b, b, a2], 2, 2);
        let result = Matrix.mul(Matrix.mul(J.transpose(), A), J);
        expect(result.isDiagonal()).toBeTruthy();
    });
});

describe('Upper triangular zeroing', () => {
    let A: Matrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        3, 4, 2, -2,
        3, 5, 1, 2], 4, 4);
    describe("QR", () => {
        const checkResult = (Q: Matrix, R: Matrix) => {
            expect(R.isTriangular(true)).toBeTruthy();
            expect(Q.isOrthogonal()).toBeTruthy();
            expect(Matrix.lInfDistance(A, Matrix.mul(Q, R))).toBeLessThan(SmallTolerance);
        };
        test('Givens rotations: implicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = 0; col < R.numCols(); ++col) {
                for (let row = R.numRows() - 1; row > col; --row) {
                    let i = row;
                    let j = col;
                    let givensCoeffs = givens(R.get(j, col), R.get(i, col));
                    applyGivensFromLeft(R, givensCoeffs, i, j);
                    applyTransposeGivensFromRight(Q, givensCoeffs, i, j);
                    expect(R.get(j, col)).toBeCloseTo(givensCoeffs.r);
                    expect(R.get(i, col)).toBeCloseTo(0);
                }
            }
            checkResult(Q, R);
        });
        // QR with givens rotations
        test('Givens rotation: explicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = 0; col < R.numCols(); ++col) {
                for (let row = R.numRows() - 1; row > col; --row) {
                    let i = row;
                    let j = col;
                    let givensCoeffs = givens(R.get(j, col), R.get(i, col));
                    let Q_k = makeGivensMatrix(givensCoeffs, R.numRows(), i, j);
                    expect(Q_k.isOrthogonal()).toBeTruthy();
                    Q = Matrix.mul(Q, Q_k.transpose());
                    R = Matrix.mul(Q_k, R);
                    expect(R.get(j, col)).toBeCloseTo(givensCoeffs.r);
                    expect(R.get(i, col)).toBeCloseTo(0);
                }
            }
            checkResult(Q, R);
        })
        test('Householder reflections: implicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = 0; col + 1 < R.numCols(); ++col) {
                let xNorm = 0.0;
                for (let row = col; row < R.numRows(); ++row)
                    xNorm += Math.pow(R.get(row, col), 2);
                xNorm = -sign(R.get(col, col)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorCol(R, col, col);
                applyHouseholderFromLeft(v, R, col);
                applyHouseholderFromRight(v, Q, col);
                expect(R.get(col, col)).toBeCloseTo(xNorm);
                for (let row = col + 1; row < R.numRows(); ++row)
                    expect(R.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, R);
        });
        test('Householder rotation: explicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = 0; col + 1 < R.numCols(); ++col) {
                let xNorm = 0.0;
                for (let row = col; row < R.numRows(); ++row)
                    xNorm += Math.pow(R.get(row, col), 2);
                xNorm = -sign(R.get(col, col)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorCol(R, col, col);
                let Q_k = makeHouseholderMatrix(v, R.numRows() - v.size(), R.numRows());
                expect(Q_k.isOrthogonal()).toBeTruthy();
                Q = Matrix.mul(Q, Q_k);
                R = Matrix.mul(Q_k, R);
                expect(R.get(col, col)).toBeCloseTo(xNorm);
                for (let row = col + 1; row < R.numRows(); ++row)
                    expect(R.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, R);
        });
    });
    describe("RQ", () => {
        const checkResult = (Q: Matrix, R: Matrix) => {
            expect(R.isTriangular(true)).toBeTruthy();
            expect(Q.isOrthogonal()).toBeTruthy();
            expect(Matrix.lInfDistance(A, Matrix.mul(R, Q))).toBeLessThan(SmallTolerance);
        };
        test('Givens rotations: implicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = R.numRows() - 1; row > 0; --row) {
                for (let col = 0; col < row; ++col) {
                    let i = row;
                    let j = col;
                    let givensCoeffs = givens(R.get(row, i), R.get(row, j));
                    applyGivensFromRight(R, givensCoeffs, i, j);
                    applyTransposeGivensFromLeft(Q, givensCoeffs, i, j);
                    expect(R.get(row, i)).toBeCloseTo(givensCoeffs.r);
                    expect(R.get(row, j)).toBeCloseTo(0);
                }
            }
            checkResult(Q, R);
        });
        test('Givens rotations: explicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = R.numRows() - 1; row > 0; --row) {
                for (let col = 0; col < row; ++col) {
                    let i = row;
                    let j = col;
                    let givensCoeffs = givens(R.get(row, i), R.get(row, j));
                    let Q_k = makeGivensMatrix(givensCoeffs, R.numRows(), i, j);
                    expect(Q_k.isOrthogonal()).toBeTruthy();
                    Q = Matrix.mul(Q_k.transpose(), Q);
                    R = Matrix.mul(R, Q_k);
                    expect(R.get(row, i)).toBeCloseTo(givensCoeffs.r);
                    expect(R.get(row, j)).toBeCloseTo(0);
                }
            }
            checkResult(Q, R);
        });
        test('Householder rotation: implicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = A.numRows() - 1; row > 0; --row) {
                let xNorm = 0.0;
                for (let col = 0; col <= row; ++col)
                    xNorm += Math.pow(R.get(row, col), 2);
                xNorm = -sign(R.get(row, row)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorRow(R, row, 0, row + 1, row);
                applyHouseholderFromRight(v, R, 0);
                applyHouseholderFromLeft(v, Q, 0);
                expect(R.get(row, row)).toBeCloseTo(xNorm);
                for (let col = 0; col < row; ++col)
                    expect(R.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, R);
        });
        test('Householder rotation: explicit', () => {
            let R = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = A.numRows() - 1; row > 0; --row) {
                let xNorm = 0.0;
                for (let col = 0; col <= row; ++col)
                    xNorm += Math.pow(R.get(row, col), 2);
                xNorm = -sign(R.get(row, row)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorRow(R, row, 0, row + 1, row);
                let Q_k = makeHouseholderMatrix(v, 0, R.numRows());
                expect(Q_k.isOrthogonal()).toBeTruthy();
                Q = Matrix.mul(Q_k, Q);
                R = Matrix.mul(R, Q_k);
                expect(R.get(row, row)).toBeCloseTo(xNorm);
                for (let col = 0; col < row; ++col)
                    expect(R.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, R);
        });
    });
});

describe('Lower triangular zeroing', () => {
    let A: Matrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        3, 4, 2, -2,
        3, 5, 1, 2], 4, 4);
    describe("LQ", () => {
        const checkResult = (Q: Matrix, L: Matrix) => {
            expect(L.isTriangular(false)).toBeTruthy();
            expect(Q.isOrthogonal()).toBeTruthy();
            expect(Matrix.lInfDistance(A, Matrix.mul(L, Q))).toBeLessThan(SmallTolerance);
        };
        test('Givens rotation: implicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = 0; row < L.numRows(); ++row) {
                for (let col = L.numCols() - 1; col > row; --col) {
                    let i = col;
                    let j = row;
                    let givensCoeffs = givens(L.get(row, j), L.get(row, i));
                    applyTransposeGivensFromRight(L, givensCoeffs, i, j);
                    applyGivensFromLeft(Q, givensCoeffs, i, j);
                    expect(L.get(row, j)).toBeCloseTo(givensCoeffs.r);
                    expect(L.get(row, i)).toBeCloseTo(0);
                }
            }
            checkResult(Q, L);
        });
        test('Givens rotation: explicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = 0; row < L.numRows(); ++row) {
                for (let col = L.numCols() - 1; col > row; --col) {
                    let i = col;
                    let j = row;
                    let givensCoeffs = givens(L.get(row, j), L.get(row, i));
                    let Q_k = makeGivensMatrix(givensCoeffs, L.numCols(), i, j);
                    expect(Q_k.isOrthogonal()).toBeTruthy();
                    L = Matrix.mul(L, Q_k.transpose());
                    Q = Matrix.mul(Q_k, Q);
                    expect(L.get(row, j)).toBeCloseTo(givensCoeffs.r);
                    expect(L.get(row, i)).toBeCloseTo(0);
                }
            }
            checkResult(Q, L);
        });
        test('Householder rotation: implicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = 0; row < L.numRows(); ++row) {
                let xNorm = 0.0;
                for (let col = row; col < L.numCols(); ++col)
                    xNorm += Math.pow(L.get(row, col), 2);
                xNorm = -sign(L.get(row, row)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorRow(L, row, row);
                applyHouseholderFromRight(v, L, row);
                applyHouseholderFromLeft(v, Q, row);
                expect(L.get(row, row)).toBeCloseTo(xNorm);
                for (let col = row + 1; col < L.numCols(); ++col)
                    expect(L.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, L);
        });
        test('Householder rotation: explicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let row = 0; row < L.numRows(); ++row) {
                let xNorm = 0.0;
                for (let col = row; col < L.numCols(); ++col)
                    xNorm += Math.pow(L.get(row, col), 2);
                xNorm = -sign(L.get(row, row)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorRow(L, row, row);
                let Q_k = makeHouseholderMatrix(v, L.numRows() - v.size(), L.numRows());
                expect(Q_k.isOrthogonal()).toBeTruthy();
                Q = Matrix.mul(Q_k, Q);
                L = Matrix.mul(L, Q_k);

                expect(L.get(row, row)).toBeCloseTo(xNorm);
                for (let col = row + 1; col < L.numCols(); ++col)
                    expect(L.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, L);
        });
    });
    describe("QL", () => {
        const checkResult = (Q: Matrix, L: Matrix) => {
            expect(L.isTriangular(false)).toBeTruthy();
            expect(Q.isOrthogonal()).toBeTruthy();
            expect(Matrix.lInfDistance(A, Matrix.mul(Q, L))).toBeLessThan(SmallTolerance);
        };
        test('Givens rotations: implicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = L.numCols() - 1; col > 0; --col) {
                for (let row = 0; row < col; ++row) {
                    let i = col;
                    let j = row;
                    let givensCoeffs = givens(L.get(i, col), L.get(j, col));
                    applyTransposeGivensFromLeft(L, givensCoeffs, i, j);
                    applyGivensFromRight(Q, givensCoeffs, i, j);
                    expect(L.get(i, col)).toBeCloseTo(givensCoeffs.r);
                    expect(L.get(j, col)).toBeCloseTo(0);
                }
            }
            checkResult(Q, L);
        });
        test('Givens rotations: explicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = L.numCols() - 1; col > 0; --col) {
                for (let row = 0; row < col; ++row) {
                    let i = col;
                    let j = row;
                    let givensCoeffs = givens(L.get(i, col), L.get(j, col));
                    let Q_k = makeGivensMatrix(givensCoeffs, L.numCols(), i, j);
                    expect(Q_k.isOrthogonal()).toBeTruthy();
                    L = Matrix.mul(Q_k.transpose(), L);
                    Q = Matrix.mul(Q, Q_k);
                    expect(L.get(i, col)).toBeCloseTo(givensCoeffs.r);
                    expect(L.get(j, col)).toBeCloseTo(0);
                }
            }
            checkResult(Q, L);
        });
        test('Householder rotation: implicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = L.numCols() - 1; col > 0; --col) {
                let xNorm = 0.0;
                for (let row = 0; row <= col; ++row)
                    xNorm += Math.pow(L.get(row, col), 2);
                xNorm = -sign(L.get(col, col)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorCol(L, 0, col, col + 1, col);
                applyHouseholderFromLeft(v, L, 0);
                applyHouseholderFromRight(v, Q, 0);

                expect(L.get(col, col)).toBeCloseTo(xNorm);
                for (let row = 0; row < col; ++row)
                    expect(L.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, L);
        });
        test('Householder rotation: explicit', () => {
            let L = A.clone();
            let Q = Matrix.identity(A.numRows());
            for (let col = L.numCols() - 1; col > 0; --col) {
                let xNorm = 0.0;
                for (let row = 0; row <= col; ++row)
                    xNorm += Math.pow(L.get(row, col), 2);
                xNorm = -sign(L.get(col, col)) * Math.sqrt(xNorm);
                let v = calcHouseholderVectorCol(L, 0, col, col + 1, col);
                let Q_k = makeHouseholderMatrix(v, 0, L.numRows());
                expect(Q_k.isOrthogonal()).toBeTruthy();
                Q = Matrix.mul(Q, Q_k);
                L = Matrix.mul(Q_k, L);

                expect(L.get(col, col)).toBeCloseTo(xNorm);
                for (let row = 0; row < col; ++row)
                    expect(L.get(row, col)).toBeCloseTo(0);
            }
            checkResult(Q, L);
        });
    });
});

describe('Upper hessenberg zeroing', () => {
    let A: Matrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        3, 4, 2, -2,
        3, 5, 1, 2], 4, 4);
    // QAQT = H;
    test('Givens rotations: explicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let col = 0; col + 2 < H.numCols(); ++col) {
            for (let row = H.numRows() - 1; row > col + 1; --row) {
                const i = row;
                const j = col + 1;
                let givensCoeffs = givens(H.get(j, col), H.get(i, col));
                let Q_k = makeGivensMatrix(givensCoeffs, H.numRows(), i, j);
                H = Matrix.mul(Matrix.mul(Q_k, H), Q_k.transpose());
                Q = Matrix.mul(Q_k, Q);
                expect(H.get(j, col)).toBeCloseTo(givensCoeffs.r);
                expect(H.get(i, col)).toBeCloseTo(0);
            }
        }
        expect(H.isHessenberg(true));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Givens rotations: implicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let col = 0; col + 2 < H.numCols(); ++col) {
            for (let row = H.numRows() - 1; row > col + 1; --row) {
                const i = row;
                const j = col + 1;
                let givensCoeffs = givens(H.get(j, col), H.get(i, col));
                applyGivensFromLeft(H, givensCoeffs, i, j);
                applyTransposeGivensFromRight(H, givensCoeffs, i, j);
                applyGivensFromLeft(Q, givensCoeffs, i, j);
                expect(H.get(j, col)).toBeCloseTo(givensCoeffs.r);
                expect(H.get(i, col)).toBeCloseTo(0);
            }
        }
        expect(H.isHessenberg(true));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Householder reflections:explicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let col = 0; col + 2 < H.numCols(); ++col) {
            let xNorm = 0.0;
            for (let row = col + 1; row < H.numRows(); ++row)
                xNorm += Math.pow(H.get(row, col), 2);
            xNorm = -sign(H.get(col + 1, col)) * Math.sqrt(xNorm);
            let v = calcHouseholderVectorCol(H, col + 1, col, H.numRows() - col - 1);
            let Q_k = makeHouseholderMatrix(v, H.numRows() - v.size(), H.numRows());
            expect(Q_k.isOrthogonal()).toBeTruthy();
            expect(Q_k.isSymmetric()).toBeTruthy();
            Q = Matrix.mul(Q_k, Q);
            H = Matrix.mul(Matrix.mul(Q_k, H), Q_k);
            expect(H.get(col + 1, col)).toBeCloseTo(xNorm);
            for (let row = col + 2; row < H.numRows(); ++row)
                expect(H.get(row, col)).toBeCloseTo(0);
        }
        expect(H.isHessenberg(true));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Householder reflections:implicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let col = 0; col + 2 < H.numCols(); ++col) {
            let xNorm = 0.0;
            for (let row = col + 1; row < H.numRows(); ++row)
                xNorm += Math.pow(H.get(row, col), 2);
            xNorm = -sign(H.get(col + 1, col)) * Math.sqrt(xNorm);
            let v = calcHouseholderVectorCol(H, col + 1, col, H.numRows() - col - 1);
            applyHouseholderFromLeft(v, H, col + 1);
            applyHouseholderFromRight(v, H, col + 1);
            applyHouseholderFromLeft(v, Q, col + 1);
            expect(H.get(col + 1, col)).toBeCloseTo(xNorm);
            for (let row = col + 2; row < H.numRows(); ++row)
                expect(H.get(row, col)).toBeCloseTo(0);
        }
        expect(H.isHessenberg(true));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
});

describe('Lower hessenberg zeroing', () => {
    let A: Matrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        3, 4, 2, -2,
        3, 5, 1, 2], 4, 4);
    test('Givens rotations: explicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let row = 0; row + 2 < H.numRows(); ++row) {
            for (let col = H.numCols() - 1; col > row + 1; --col) {
                const i = col;
                const j = row + 1;
                let givensCoeffs = givens(H.get(row, j), H.get(row, i));
                let Q_k = makeGivensMatrix(givensCoeffs, H.numRows(), i, j);
                H = Matrix.mul(Matrix.mul(Q_k, H), Q_k.transpose());
                Q = Matrix.mul(Q_k, Q);
                expect(H.get(row, j)).toBeCloseTo(givensCoeffs.r);
                expect(H.get(row, i)).toBeCloseTo(0);
            }
        }
        expect(H.isHessenberg(false));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Givens rotations: implicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let row = 0; row + 2 < H.numRows(); ++row) {
            for (let col = H.numCols() - 1; col > row + 1; --col) {
                const i = col;
                const j = row + 1;
                let givensCoeffs = givens(H.get(row, j), H.get(row, i));
                applyGivensFromLeft(H, givensCoeffs, i, j);
                applyTransposeGivensFromRight(H, givensCoeffs, i, j);
                applyGivensFromLeft(Q, givensCoeffs, i, j);
                expect(H.get(row, j)).toBeCloseTo(givensCoeffs.r);
                expect(H.get(row, i)).toBeCloseTo(0);
            }
        }
        expect(H.isHessenberg(false));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Householder reflections:explicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let row = 0; row + 2 < H.numRows(); ++row) {
            let xNorm = 0.0;
            for (let col = row + 1; col < H.numRows(); ++col)
                xNorm += Math.pow(H.get(row, col), 2);
            xNorm = -sign(H.get(row, row + 1)) * Math.sqrt(xNorm);
            let v = calcHouseholderVectorRow(H, row, row + 1);
            let Q_k = makeHouseholderMatrix(v, H.numCols() - v.size(), H.numCols());
            expect(Q_k.isOrthogonal()).toBeTruthy();
            expect(Q_k.isSymmetric()).toBeTruthy();
            Q = Matrix.mul(Q_k, Q);
            H = Matrix.mul(Matrix.mul(Q_k, H), Q_k);
            expect(H.get(row, row + 1)).toBeCloseTo(xNorm);
            for (let col = row + 2; col < H.numRows(); ++col)
                expect(H.get(row, col)).toBeCloseTo(0);
        }
        expect(H.isHessenberg(false));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
    test('Householder reflections:implicit', () => {
        let H = A.clone();
        let Q = Matrix.identity(H.numRows());
        for (let row = 0; row + 2 < H.numRows(); ++row) {
            let xNorm = 0.0;
            for (let col = row + 1; col < H.numRows(); ++col)
                xNorm += Math.pow(H.get(row, col), 2);
            xNorm = -sign(H.get(row, row + 1)) * Math.sqrt(xNorm);
            let v = calcHouseholderVectorRow(H, row, row + 1);
            applyHouseholderFromLeft(v, H, row + 1);
            applyHouseholderFromRight(v, H, row + 1);
            applyHouseholderFromLeft(v, Q, row + 1);
            expect(H.get(row, row + 1)).toBeCloseTo(xNorm);
            for (let col = row + 2; col < H.numRows(); ++col)
                expect(H.get(row, col)).toBeCloseTo(0);
        }
        expect(H.isHessenberg(false));
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(H, Matrix.mul(Matrix.mul(Q, A), Q.transpose()))).toBeLessThan(SmallTolerance);
    });
});

describe('Hessenbergization', () => {
});

describe('Tridiagonalization', () => {
    let A = new Matrix([
        4, 1, -2, 2,
        1, 2, 0, 1,
        -2, 0, 3, -2,
        2, 1, -2, -1], 4, 4);
    // test symmetric householder by generating hessenberg matrix QAQT = H
    test('Householder', () => {
        let Q = Matrix.identity(A.numRows());
        let T = A.clone();
        for (let iter = 0; iter + 2 < T.numCols(); ++iter) {
            let v = calcHouseholderVectorRow(T, iter, iter + 1);
            applyHouseholderFromLeft(v, Q, iter + 1);
            applyHouseholderFromLeft(v, T, iter + 1);
            applyHouseholderFromRight(v, T, iter + 1);
            // check matrices
        }
        expect(T.isHessenberg()).toBeTruthy();
        expect(T.isTridiagonal()).toBeTruthy();
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(Matrix.mul(Q, Matrix.mul(A, Q.transpose())), T)).toBeLessThan(SmallTolerance);
    });
});

//v8Profiler.setGenerateType(1);
const title = 'Hessenberg-performance';

function startPerformanceProfiling() {
    /*v8Profiler.startProfiling(title, true);
    afterAll(() => {
        const profile = v8Profiler.stopProfiling(title);
        profile.export(function (error, result: any) {
            // if it doesn't have the extension .cpuprofile then
            // chrome's profiler tool won't like it.
            // examine the profile:
            //   Navigate to chrome://inspect
            //   Click Open dedicated DevTools for Node
            //   Select the profiler tab
            //   Load your file
            fs.writeFileSync(`${title}.cpuprofile`, result);
            profile.delete();
        });
    });*/
}

describe.skip('Hessenberg performance', () => {
    const numRepetitions = 20;
    let matrices: Matrix[] = [];
    for (const size of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        matrices.push(Matrix.random(size, size));

    // startPerformanceProfiling();

    test('Hessenberg partial default', () => {
        let stopWatch = new StopWatch();
        let time: number[] = [];
        for (const A of matrices) {
            let H: Matrix = A;
            stopWatch.reset();
            for (let i = 0; i < numRepetitions; ++i)
                H = makeHessenberg(A, undefined);
            time.push(stopWatch.elapsed() / numRepetitions);
            expect(H.isHessenberg(true)).toBeTruthy();
        }
        console.log(`Hessenberg partial default: ${time}`);
    });
    test('Full Hessenberg default', () => {
        let stopWatch = new StopWatch();
        let time: number[] = [];
        for (const A of matrices) {
            let Q: Matrix = Matrix.empty(A.width(), A.width());
            let H: Matrix = A;
            stopWatch.reset();
            for (let i = 0; i < numRepetitions; ++i)
                H = makeHessenberg(A, Q);
            time.push(stopWatch.elapsed() / numRepetitions);
            expect(H.isHessenberg(true)).toBeTruthy();
            expect(Q.isOrthogonal()).toBeTruthy();
            expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, A), Q.transpose()), H)).toBeLessThan(SmallTolerance);
        }
        console.log(`Full hessenberg default: ${time}`);
    });
});

test.skip('Triangular alt', () => {
    let S: Matrix = new Matrix([
        4, 1, -2, 2,
        1, 2, 0, 1,
        -2, 0, 3, -2,
        2, 1, -2, -1], 4, 4);
    let expectedQ = new Matrix([
        1, 0, 0, 0,
        0, -1 / 3, 2 / 3, -2 / 3,
        0, 2 / 15, -2 / 3, -11 / 15,
        0, -14 / 15, -1 / 3, 2 / 15
    ], 4, 4);
    let expectedT = new Matrix([
        4, -3, 0, 0,
        -3, 10 / 3, -5 / 3, 0,
        0, -5 / 3, -33 / 25, 68 / 75,
        0, 0, 68 / 75, 149 / 75], 4, 4);
    let Q: Matrix = Matrix.empty(4, 4);
    let T1 = makeTridiagonalAlt(S);
    let T = makeTridiagonalAlt(S, Q);
    let TOld = makeTridiagonal(S);
    console.log(`Expected: ${expectedT.toString()}`);
    console.log(`Told: ${TOld.toString()}`);
    console.log(`TAlt1: ${T1.toString()}`);
    console.log(`TAlt2: ${T.toString()}`);
    expect(Matrix.lInfDistance(T, T1)).toBeLessThan(SmallTolerance);
    expect(Matrix.lInfDistance(TOld, T)).toBeLessThan(SmallTolerance);
    expect(T.isHessenberg(true)).toBeTruthy();
    expect(T.isTridiagonal()).toBeTruthy();
    expect(Q.isOrthogonal()).toBeTruthy();
    expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, S), Q.transpose()), T)).toBeLessThan(SmallTolerance);
    expect(Matrix.lInfDistance(T, expectedT)).toBeLessThan(SmallTolerance);
    expect(Matrix.lInfDistance(Q, expectedQ)).toBeLessThan(SmallTolerance);
});

describe.skip('Triangular performance', () => {
    const generator = new MatrixGenerator(new JSGenerator());
    const numRepetitions = 10;
    let matrices: Matrix[] = [];
    for (const size of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        matrices.push(generator.randomSymmetric(size));

    startPerformanceProfiling();

    test('Partial Triangular alt', () => {
        //let stopWatch = new StopWatch();
        //let time: number[] = [];
        for (const S of matrices) {
            //stopWatch.reset();
            let T: Matrix = S;
            for (let i = 0; i < numRepetitions; ++i)
                T = makeTridiagonalAlt(S, undefined);
            //time.push(stopWatch.elapsed());
            //expect(T.isTridiagonal()).toBeTruthy();
        }
        //console.log(`Partial Triangular alt' Q: ${time}`);
    });
    test('Full Triangular alt', () => {
        //let stopWatch = new StopWatch();
        //let time: number[] = [];
        for (const S of matrices) {
            let Q: Matrix = Matrix.empty(S.width(), S.width());
            //stopWatch.reset();
            let T: Matrix = S;
            for (let i = 0; i < numRepetitions; ++i) {
                T = makeTridiagonalAlt(S, Q);
            }
            //time.push(stopWatch.elapsed());
            //expect(T.isTridiagonal()).toBeTruthy();
            //expect(Q.isOrthogonal()).toBeTruthy();
            //expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, S), Q.transpose()), T)).toBeLessThan(SmallTolerance);
        }
        //console.log(`Full Triangular alt Q: ${time}`);
    });
    test('Partial Triangular default', () => {
        //let stopWatch = new StopWatch();
        //let time: number[] = [];
        for (const S of matrices) {
            //stopWatch.reset();
            let T: Matrix = S;
            for (let i = 0; i < numRepetitions; ++i)
                T = makeTridiagonal(S, undefined);
            //time.push(stopWatch.elapsed());
            //expect(T.isTridiagonal()).toBeTruthy();
        }
        //console.log(`Partial Triangular default: ${time}`);
    });
    test('Full Triangular default', () => {
        //let stopWatch = new StopWatch();
        //let time: number[] = [];
        for (const S of matrices) {
            let Q: Matrix = Matrix.empty(S.width(), S.width());
            //stopWatch.reset();
            let T: Matrix = S;
            for (let i = 0; i < numRepetitions; ++i)
                T = makeTridiagonal(S, Q);
            //time.push(stopWatch.elapsed());
            //expect(T.isTridiagonal()).toBeTruthy();
            //expect(Q.isOrthogonal()).toBeTruthy();
            //expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, S), Q.transpose()), T)).toBeLessThan(SmallTolerance);

        }
        //console.log(`Full Triangular default: ${time}`);
    });
});

describe('Hessenberg form decomposition', () => {
    test('Tridiagonal', () => {
        let A: Matrix = new Matrix([
            4, 1, -2, 2,
            1, 2, 0, 1,
            -2, 0, 3, -2,
            2, 1, -2, -1], 4, 4);
        let expectedQ = new Matrix([
            1, 0, 0, 0,
            0, -1 / 3, 2 / 3, -2 / 3,
            0, 2 / 15, -2 / 3, -11 / 15,
            0, -14 / 15, -1 / 3, 2 / 15
        ], 4, 4);
        let expectedH = new Matrix([
            4, -3, 0, 0,
            -3, 10 / 3, -5 / 3, 0,
            0, -5 / 3, -33 / 25, 68 / 75,
            0, 0, 68 / 75, 149 / 75], 4, 4);
        let Q: Matrix = Matrix.empty(4, 4);
        let H = makeHessenberg(A, Q);
        expect(H.isHessenberg(true)).toBeTruthy();
        expect(H.isTridiagonal()).toBeTruthy();
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, A), Q.transpose()), H)).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(H, expectedH)).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(Q, expectedQ)).toBeLessThan(SmallTolerance);
        Q = Matrix.empty(4, 4);
        let T = makeTridiagonal(A, Q);
        expect(T.isTridiagonal()).toBeTruthy();
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(T, expectedH)).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(Q, expectedQ)).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, A), Q.transpose()), T)).toBeLessThan(SmallTolerance);
    });
    const testData: Matrix[] = [
        new Matrix([
            1, 2, 3, 4,
            5, 6, 7, 8,
            3, 4, 2, -2,
            3, 5, 1, 2], 4, 4),
        new Matrix([
            0, -0.5, 0, 0,
            0, 0.25, -0.5, 0,
            0, -0.125, 0.25, -0.5,
            0, 0.0625, -0.125, 0.250], 4, 4),
        new Matrix([
            -0.5, 0, 0, 0,
            0.25, 0, -0.5, 0,
            -0.125, 0, 0.25, -0.5,
            0.0625, 0, -0.125, 0.250], 4, 4),
        new Matrix([
            0, -0.5, 0, 0,
            -0.5, 0.25, 0, 0,
            0.25, -0.125, 0, -0.5,
            -0.125, 0.0625, 0, 0.250], 4, 4),
        new Matrix([
            0, -0.5, 0, 0,
            0, 0.25, -0.5, 0,
            -0.5, -0.125, 0.25, 0,
            0.250, 0.0625, -0.125, 0], 4, 4)
    ];
    test.each(testData)('Hessenberg %#', (A: Matrix) => {
        let Q: Matrix = Matrix.empty(4, 4);
        let H = makeHessenberg(A, Q);
        expect(H.isHessenberg(true)).toBeTruthy();
        expect(Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(Matrix.mul(Matrix.mul(Q, A), Q.transpose()), H)).toBeLessThan(SmallTolerance);
    });
});

describe.skip('Triangulization', () => {
    const data: Matrix[] = [new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        3, 4, 2, -2,
        3, 5, 1, 2], 4, 4),
    new Matrix([
        0, 2, 0, 4,
        0, 6, 0, 8,
        0, 4, 0, -2,
        0, 5, 0, 2], 4, 4),
    new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 4, 6, 8,
        5, 6, 7, 8], 4, 4),
    new Matrix([
        1, 2, 3, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
        5, 6, 7, 8], 4, 4)];
    describe.each(data)("Square %#", (matrix: Matrix) => {
        interface Method {
            method: ZeroingMethod, name: String
        };
        describe.each([{ method: ZeroingMethod.Givens, name: "Givens" }, { method: ZeroingMethod.Housholder, name: "Housholder" }])("Zeroing method %#",
            (method: Method) => {
                test("QR", () => {
                    let decomposition = new OrthogonalDecomposition(matrix, method.method, false, OrthogonalDecompositionType.QR);
                    expect(decomposition.Q).not.toBeNull();
                    expect(decomposition.Q.isOrthogonal()).toBeTruthy();
                    expect(decomposition.T.isTriangular(true)).toBeTruthy();
                    expect(Matrix.lInfDistance(Matrix.mul(decomposition.Q, decomposition.T), matrix)).toBeLessThan(Tolerance);
                });
                test("RQ", () => {
                    let decomposition = new OrthogonalDecomposition(matrix, method.method, false, OrthogonalDecompositionType.RQ);
                    expect(decomposition.Q).not.toBeNull();
                    expect(decomposition.Q.isOrthogonal()).toBeTruthy();
                    expect(decomposition.T.isTriangular(true)).toBeTruthy();
                    expect(Matrix.lInfDistance(Matrix.mul(decomposition.T, decomposition.Q), matrix)).toBeLessThan(Tolerance);
                });
                test("QL", () => {
                    let decomposition = new OrthogonalDecomposition(matrix, method.method, false, OrthogonalDecompositionType.QL);
                    expect(decomposition.Q).not.toBeNull();
                    expect(decomposition.Q.isOrthogonal()).toBeTruthy();
                    expect(decomposition.T.isTriangular(false)).toBeTruthy();
                    expect(Matrix.lInfDistance(Matrix.mul(decomposition.Q, decomposition.T), matrix)).toBeLessThan(Tolerance);
                });
                test("LQ", () => {
                    let decomposition = new OrthogonalDecomposition(matrix, method.method, false, OrthogonalDecompositionType.LQ);
                    expect(decomposition.Q).not.toBeNull();
                    expect(decomposition.Q.isOrthogonal()).toBeTruthy();
                    expect(decomposition.T.isTriangular(false)).toBeTruthy();
                    expect(Matrix.lInfDistance(Matrix.mul(decomposition.T, decomposition.Q), matrix)).toBeLessThan(Tolerance);
                });
            });
    });
})