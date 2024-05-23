import { complex } from "../complex";
import Matrix from "./denseMatrix";
import RandomNumberGenerator from "../random/generator";
import { randomNormalDistr } from "../random/utils";
import { QR, ZeroingMethod } from "../solvers/linear systems/qr";
import { randomArray, sign } from "../utils";
import { binomial } from "../utils";

export enum SignType {
    Positive,
    Negative,
    Random,
    Ignore
}

export class MatrixGenerator {
    protected generator: RandomNumberGenerator;
    constructor(generator: RandomNumberGenerator) {
        this.generator = generator;
    }
    random(numRows: number, numCols: number, min: number = 0.0, max: number = 1.0): Matrix {
        let data = [];
        for (let i = 0; i < numRows * numCols; i++)
            data.push(this.generator.random(min, max));
        return new Matrix(data, numRows, numCols);
    }
    randomSymmetric(size: number, min: number = 0.0, max: number = 1.0): Matrix {
        let matrix = Matrix.empty(size, size);
        for (let i = 0; i < size; ++i) {
            for (let j = i; j < size; ++j) {
                let value = this.generator.random(min, max);
                matrix.set(i, j, value);
                matrix.set(j, i, value)
            }
        }
        return matrix;
    }
    randomOrthogonal(size: number): Matrix {
        let m = Matrix.generate(size, size, (r: number, c: number) => { return randomNormalDistr(this.generator); });
        const qrSolver = new QR(m, ZeroingMethod.Housholder, false);
        let result = qrSolver.Q;
        for (let row = 0; row < size; ++row) {
            for (let col = 0; col < size; ++col)
                result.set(row, col, result.get(row, col) * sign(qrSolver.R.get(col, col)));
        }
        return result;
    }
    randomDiagonal(size: number, shift: number, min: number = 0.0, max: number = 1.0): Matrix {
        let matrix = Matrix.empty(size, size);
        for (let row = 0; row < size; ++row) {
            for (let col = Math.max(0, row - shift); col <= Math.min(size - 1, row + shift); ++col)
                matrix.set(row, col, this.generator.random(min, max));
        }
        return matrix;
    }
    randomWithEigenvalues(size: number, minEig: number, maxEig: number, makeSymmetric: boolean, sign: SignType): Matrix {
        let eigenvalues = randomArray(size, minEig, maxEig);
        if (sign != SignType.Ignore) {
            for (let i = 0; i < size; ++i) {
                let value = Math.abs(eigenvalues[i]);
                switch (sign) {
                    case SignType.Negative:
                        eigenvalues[i] = -value;
                        break;
                    case SignType.Positive:
                        eigenvalues[i] = value;
                        break;
                    case SignType.Random:
                        eigenvalues[i] = Math.sign(this.generator.randomUnit() - 0.5) * value;
                }
            }
        }
        return this.randomFromEigenvalues(eigenvalues, makeSymmetric);
    }
    randomFromEigenvalues(eigenvalues: number[], makeSymmetric: boolean): Matrix {
        let M = Matrix.diag(eigenvalues);
        if (!makeSymmetric) {
            for (let row = 0; row + 1 < M.numRows(); ++row) {
                for (let col = row + 1; col < M.numCols(); ++col)
                    M.set(row, col, this.generator.randomUnit() * 2 - 1);
            }
        }
        let Q = this.randomOrthogonal(eigenvalues.length);
        return Matrix.mul(Matrix.mul(Q, M), Q.transpose());
    }
    /**
     * Generate matrix from list of eigenvalues, if eigenvalue is complex it will be treated as conjugate pairs,
     * if it's real it will be counted as one eigenvalue
     * @param eigenvalues 
     */
    randomFromComplexPairsEigenvalues(eigenvalues: complex[]): Matrix {
        let size = 0;
        for (const eigenvalue of eigenvalues) {
            size++;
            if (Math.abs(eigenvalue.y) != 0)
                size++;
        }
        let M = Matrix.empty(size, size);
        let counter = 0;
        for (const eigenvalue of eigenvalues) {
            if (Math.abs(eigenvalue.y) != 0) {
                M.set(counter, counter, eigenvalue.x);
                M.set(counter, counter + 1, -eigenvalue.y);
                M.set(counter + 1, counter, eigenvalue.y);
                M.set(counter + 1, counter + 1, eigenvalue.x);
                counter += 2;
            } else {
                M.set(counter, counter, eigenvalue.x);
                counter++;
            }
        }
        let Q = this.randomOrthogonal(size);
        return Matrix.mul(Matrix.mul(Q, M), Q.transpose());
    }

    static hilbertMatrix(size: number): Matrix {
        let m = Matrix.empty(size, size);
        for (let i = 0; i < size; ++i) {
            for (let j = 0; j <= i; ++j) {
                let value = 1 / (i + j + 1);
                m.set(j, i, value);
                m.set(i, j, value)
            }
        }
        return m;
    }

    static inverseHilbertMatrix(size: number): Matrix {
        let m = Matrix.empty(size, size);
        for (let i = 0; i < size; ++i) {
            for (let j = 0; j <= i; ++j) {
                let value = (i + j) & 1 ? -1 : 1;
                value *= i + j + 1;
                value *= binomial(size + i, size - j - 1);
                value *= binomial(size + j, size - i - 1);
                value *= Math.pow(binomial(i + j, i), 2);
                m.set(j, i, value);
                m.set(i, j, value)
            }
        }
        return m;
    }
}