import { complex } from "../complex";
import Matrix from "../dense/denseMatrix";
import JSGenerator from "../random/js";
import { calcEigenvalues } from "../solvers/linear systems/eigenvalues";
import { SmallTolerance, Tolerance } from "../utils";
import { MatrixGenerator } from "./matrixGenerator";

function checkEigenvalues(expected: number[], actual: number[], numDigits: number = 2): void {
    expect(expected.length).toBe(actual.length);
    expected.sort((a: number, b: number) => a - b);
    actual.sort((a: number, b: number) => a - b);
    for (let i = 0; i < expected.length; ++i)
        expect(expected[i]).toBeCloseTo(actual[i], numDigits);
}

function checkComplexEigenvalues(expected: complex[], actual: number[], numDigits: number = 2): void {
    let expectedTotal: number[] = [];
    for (const eigenvalue of expected) {
        if (Math.abs(eigenvalue.y) != 0) {
            expectedTotal.push(eigenvalue.x);
            expectedTotal.push(eigenvalue.x);
        }
        else { expectedTotal.push(eigenvalue.x); }
    }
    expect(expectedTotal.length).toBe(actual.length);
    actual.sort((a: number, b: number) => a - b);
    expectedTotal.sort((a: number, b: number) => a - b);
    for (let i = 0; i < expectedTotal.length; ++i)
        expect(expectedTotal[i]).toBeCloseTo(actual[i], numDigits);
}

describe('Random matrix generation tests', () => {
    let generator = new MatrixGenerator(new JSGenerator());
    const MatrixSize = 5;
    test('Random eigenvalues symmetric', () => {
        const eigenvalues: number[] = [1, 2, 3, 4, 5, -1];
        let symmetric = generator.randomFromEigenvalues(eigenvalues, true);
        expect(symmetric.isSymmetric()).toBeTruthy();
        expect(symmetric.isSquare()).toBeTruthy();
        expect(symmetric.numCols() == eigenvalues.length);

        let matEigenvalues = calcEigenvalues(symmetric, 10, Tolerance);
        checkEigenvalues(matEigenvalues, eigenvalues);
    });
    test('Random eigenvlaues general', () => {
        const eigenvalues: number[] = [1, 2, 3, 4, 5, -1];
        let matrix = generator.randomFromEigenvalues(eigenvalues, false);
        expect(matrix.isSquare()).toBeTruthy();
        expect(matrix.numCols() == eigenvalues.length);

        let matEigenvalues = calcEigenvalues(matrix, 10, Tolerance);
        checkEigenvalues(eigenvalues, matEigenvalues);
    });
    test('Random eigenvlaues complex', () => {
        const complexEigenvalues: complex[] = [new complex(1, 0), new complex(1, 0),
        new complex(2, -2), new complex(3, 0), new complex(4, 1)];
        let matrix = generator.randomFromComplexPairsEigenvalues(complexEigenvalues);
        expect(matrix.isSquare()).toBeTruthy();
        expect(matrix.numCols() == 7);

        let matEigenvalues = calcEigenvalues(matrix, 10, Tolerance);
        checkComplexEigenvalues(complexEigenvalues, matEigenvalues);
    });
    test('Random orthogonal', () => {
        let matrix = generator.randomOrthogonal(MatrixSize);
        expect(matrix.isOrthogonal()).toBeTruthy();
    });
    test('Random symmetric', () => {
        let matrix = generator.randomSymmetric(MatrixSize);
        expect(matrix.isSymmetric()).toBeTruthy();
    });
    test('Random diagonal', () => {
        let matrix = generator.randomDiagonal(MatrixSize, 0);
        expect(matrix.isDiagonal()).toBeTruthy();
    });
});

describe('Special matrix generators', () => {
    test('Hilbert matrix', () => {
        const hilbert = new Matrix(
            [
                1, 1 / 2, 1 / 3, 1 / 4, 1 / 5,
                1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6,
                1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7,
                1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8,
                1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9], 5, 5);
        const invHilbert = new Matrix(
            [
                25, -300, 1050, -1400, 630,
                -300, 4800, -18900, 26880, -12600,
                1050, -18900, 79380, -117600, 56700,
                -1400, 26880, -117600, 179200, -88200,
                630, -12600, 56700, -88200, 44100], 5, 5);
        expect(Matrix.lInfDistance(MatrixGenerator.inverseHilbertMatrix(5), invHilbert)).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(MatrixGenerator.hilbertMatrix(5), hilbert)).toBeLessThan(SmallTolerance);
    });
});