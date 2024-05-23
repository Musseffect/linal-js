
import Matrix from "../../dense/denseMatrix";
import { Tolerance, SmallTolerance, assert, SmallestTolerance, near } from "../../utils";
import Vector from "../../dense/vector";
import * as linSolvers from "./exports";
import { TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import MatrixSpectrum, { } from "../../dense/matrixSpectrum";

interface TestCase {
    m: Matrix;
    rhs: Vector;
    exactSolution: Vector;
    inverse: Matrix;
    determinant: number;
}

interface Tests {
    posDef: TestCase[],
    general: TestCase[]
};

const squareSystemTestCases: Tests = { posDef: [], general: [] };

(function () {
    squareSystemTestCases.posDef.push({
        m: new Matrix([10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8], 4, 4),
        rhs: new Vector([6, 25, -11, 15]),
        exactSolution: new Vector([1, 2, -1, 1]),
        inverse: new Matrix([259 / 2465, 23 / 2465, -3 / 145, -3 / 493, 23 / 2465, 758 / 7395, 2 / 435, -56 / 1479, -3 / 145, 2 / 435, 46 / 435, 1 / 87, -3 / 493, -56 / 1479, 1 / 87, 208 / 1479], 4, 4),
        determinant: 7395
    });
    squareSystemTestCases.posDef.push({
        m: new Matrix([1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 3, 1, 2, 3, 4], 4, 4),
        rhs: new Vector([-4, -7, -9, -10]),
        exactSolution: new Vector([-1, -1, -1, -1]),
        inverse: new Matrix([2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 1], 4, 4),
        determinant: 1
    });
    squareSystemTestCases.posDef.push({
        m: new Matrix([4, 12, -16, 12, 37, -43, -16, -43, 98], 3, 3),
        rhs: new Vector([76, 215, -396]),
        exactSolution: new Vector([1, 2, -3]),
        inverse: new Matrix([1777 / 36, -122 / 9, 19 / 9, -122 / 9, 34 / 9, -5 / 9, 19 / 9, -5 / 9, 1 / 9], 3, 3),
        determinant: 36
    });
    squareSystemTestCases.general.push({
        m: new Matrix([0.02, 0.01, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 100, 200], 4, 4),
        rhs: new Vector([0.02, 1, 4, 800]),
        exactSolution: new Vector([1, 0, 0, 4]),
        inverse: new Matrix([80, -0.6, 0.4, -0.002, -60, 1.2, -0.8, 0.004, 40, -0.8, 1.2, -0.006, -20, 0.4, -0.6, 0.008], 4, 4),
        determinant: 5
    });
    squareSystemTestCases.general.push({
        m: new Matrix([0, 1, 1, 1], 2, 2),
        rhs: new Vector([2, -1]),
        exactSolution: new Vector([-3, 2]),
        inverse: new Matrix([-1, 1, 1, 0], 2, 2),
        determinant: -1
    });
    squareSystemTestCases.general.push({
        m: new Matrix([2, 3, 5, 7], 2, 2),
        rhs: new Vector([11, 13]),
        exactSolution: new Vector([-38, 29]),
        inverse: new Matrix([-7, 3, 5, -2], 2, 2),
        determinant: -1
    });
    const checkTest = (test: TestCase) => {
        assert(test.m.isSquare(), "Expected square matrix");
        assert(test.inverse.isSquare(), "Expected square inverse");
        assert(test.exactSolution.size() == test.m.numCols(), "Inconsistent solution size");
        assert(test.rhs.size() == test.m.numRows(), "Inconsistent rhs size");
        assert(test.m.numRows() == test.inverse.numRows(), "Inconsistent inverse size");
        assert(Vector.near(test.rhs, Matrix.postMulVec(test.m, test.exactSolution), SmallestTolerance), "Incorrect solution");
        assert(Vector.near(test.exactSolution, Matrix.postMulVec(test.inverse, test.rhs), SmallestTolerance), "Incorrect inverse");
    };
    for (const test of squareSystemTestCases.general) {
        checkTest(test);
    }
    for (const test of squareSystemTestCases.posDef) {
        assert(test.m.isSymmetric(), "Expected symmetric matrix");
        assert(test.inverse.isSymmetric(), "Expected symmetric inverse");
        checkTest(test);
        squareSystemTestCases.general.push(test);
    }
})();

let id = 0;
describe.only.each(squareSystemTestCases.general)('General matrices %#', (testCase: TestCase) => {
    console.log(`Test case ${id}`);
    console.log(testCase.m.toString());
    console.log(`Diagonally dominant ${testCase.m.isDiagonallyDominant()}`);
    console.log(`Symmetric ${testCase.m.isSymmetric()}`);
    console.log(`Matrix spectral radius ${id} ${MatrixSpectrum.spectralRadius(testCase.m)}`);
    console.log(`Matrix condition number ${id} ${MatrixSpectrum.conditionNumber(testCase.m)}`);
    /*let m = new TriMatrixView(testCase.m, TriMatrixType.lower).toMatrix().inverse();
    console.log(m.toString());
    let n = Matrix.sub(Matrix.empty(testCase.m.numRows(), testCase.m.numRows()), new TriMatrixView(testCase.m, TriMatrixType.upper, 0).toMatrix());
    let mn = Matrix.mul(m, n);
    console.log(mn.toString());
    console.log(`M-1 N spectral radius ${id} ${MatrixSpectrum.spectralRadius(mn)}`);*/
    ++id;
    test.skip('gauss-zeidel-reorder', () => {
        let solver = new linSolvers.gaussSeidel();
        solver.reorderRows = true;
        solver.maxIterations = 60;
        solver.tolerance = SmallTolerance;
        let result = solver.solve(testCase.m, testCase.rhs);
        expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
    });
    test.skip('gauss-zeidel', () => {
        let result = linSolvers.gaussSeidel.solve(testCase.m, testCase.rhs, 60, SmallTolerance);
        expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
    });
    test.skip('jacobi', () => {
        let result = linSolvers.jacobi.solve(testCase.m, testCase.rhs, 60, SmallTolerance);
        expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
    });
    test.skip('sor', () => {
        let result = linSolvers.sor.solve(testCase.m, testCase.rhs, 60, 1.0, SmallTolerance);
        expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
    });
});

describe.skip('Linear solvers (dense square matrices)', () => {
    describe.skip.each(squareSystemTestCases.posDef)('Symmetric positive definite matrices %#', (testCase: TestCase) => {
        expect(testCase.m.isSymmetric()).toBeTruthy();
        describe('Factorizations', () => {
            test('LL', () => {
                let solver = new linSolvers.LLT(null, SmallTolerance);
                expect(() => solver.factorize(testCase.m)).not.toThrow();
                expect(solver.LLT).not.toBeNull();
                expect(Vector.sub(solver.solve(testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
                expect(Matrix.lInfDistance(testCase.inverse, solver.inverse() as Matrix)).toBeLessThan(SmallTolerance);
                expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);
            });
            test('LDL', () => {
                let solver = new linSolvers.LDLT(null, SmallTolerance);
                expect(() => solver.factorize(testCase.m)).not.toThrow();
                expect(solver.LDLT).not.toBeNull();
                expect(Vector.sub(solver.solve(testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
                expect(Matrix.lInfDistance(testCase.inverse, solver.inverse() as Matrix)).toBeLessThan(SmallTolerance);
                expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);
            });
        });
        describe('Iterative', () => {
            test('ConjGrad', () => {
                let result = linSolvers.CG.solve(testCase.m, testCase.rhs, 20);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(SmallTolerance);
            });
        });
    });
    describe.each(squareSystemTestCases.general)('General matrices %#', (testCase: TestCase) => {
        describe.skip('Factorizations', () => {
            test('PartialPivLU', () => {
                let solver = new linSolvers.PartialPivLU(null);
                expect(() => solver.factorize(testCase.m)).not.toThrow();
                expect(solver.LU).not.toBeNull();
                expect(Vector.sub(solver.solve(testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
                expect(Matrix.lInfDistance(testCase.inverse, solver.inverse() as Matrix)).toBeLessThan(SmallTolerance);
                expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);
                // test static variant
                expect(Vector.sub(linSolvers.PartialPivLU.solve(testCase.m, testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
            });
            test('FullPivLU', () => {
                let solver = new linSolvers.FullPivLU(null);
                expect(() => solver.factorize(testCase.m)).not.toThrow();
                expect(solver.LU).not.toBeNull();
                expect(Vector.sub(solver.solve(testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
                expect(Matrix.lInfDistance(testCase.inverse, solver.inverse() as Matrix)).toBeLessThan(SmallTolerance);
                expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);// test static variant
                expect(Vector.sub(linSolvers.FullPivLU.solve(testCase.m, testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
            });
            test('QR', () => {
                for (const method of [linSolvers.ZeroingMethod.Givens, linSolvers.ZeroingMethod.Housholder]) {
                    let solver = new linSolvers.QR(null);
                    solver.zeroingMethod = method;
                    for (const isCompact of [false, true]) {
                        solver.makeCompact = isCompact;
                        expect(() => solver.factorize(testCase.m)).not.toThrow();
                        expect(solver.Q).not.toBeNull();
                        expect(solver.R).not.toBeNull();
                        expect(solver.Q.isOrthogonal()).toBeTruthy();
                        expect(solver.R.isTriangular(true)).toBeTruthy();
                        expect(Vector.sub(solver.solve(testCase.rhs) as Vector, testCase.exactSolution).lInfNorm()).toBeLessThanOrEqual(SmallTolerance);
                        expect(Matrix.lInfDistance(testCase.inverse, solver.inverse() as Matrix)).toBeLessThan(SmallTolerance);
                        expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);
                    }
                }
            });
        });
        describe('Iterative', () => {
            test('gauss-zeidel-reorder', () => {
                let solver = new linSolvers.gaussSeidel();
                solver.reorderRows = true;
                solver.maxIterations = 60;
                solver.tolerance = SmallTolerance;
                let result = solver.solve(testCase.m, testCase.rhs);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
            test('gauss-zeidel', () => {
                let result = linSolvers.gaussSeidel.solve(testCase.m, testCase.rhs, 60, SmallTolerance);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
            test('jacobi', () => {
                let result = linSolvers.jacobi.solve(testCase.m, testCase.rhs, 60, SmallTolerance);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
            test('sor', () => {
                let result = linSolvers.sor.solve(testCase.m, testCase.rhs, 60, 1.0, SmallTolerance);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
        });
    });
});

interface RectSystemTestCase {
    matrix: Matrix;
    pseudoInverse: Matrix;
    rhs: Vector;
    leastSquaresSolution: Vector;
}

const rectSystemTestCases: RectSystemTestCase[] = [];

(function () {
    rectSystemTestCases.push({
        matrix: new Matrix([
            1, 2, 4,
            4, 5, 6,
            1, 1, 1,
            3, 2, 5,
            4, 2, 1
        ], 5, 3),
        pseudoInverse: new Matrix([
            -297 / 2692, -393 / 2692, -11 / 2692, 142 / 673, 717 / 2692,
            43 / 2692, 1063 / 2692, 201 / 2692, -331 / 673, -131 / 2692,
            289 / 2692, -243 / 2692, -89 / 2692, 170 / 673, -317 / 2692
        ], 3, 5),
        rhs: new Vector([0, 2, 10, 2, 0]),
        leastSquaresSolution: new Vector([60 / 673, 372 / 673, -4 / 673])

    });
    rectSystemTestCases.push({
        matrix: rectSystemTestCases[0].matrix.transpose(),
        pseudoInverse: rectSystemTestCases[0].pseudoInverse.transpose(),
        rhs: new Vector([1, 2, 3]),
        leastSquaresSolution: new Vector([
            164 / 673,
            251 / 673,
            31 / 673,
            -10 / 673,
            -124 / 673
        ])
    });

    for (const test of rectSystemTestCases) {
        assert(!test.matrix.isSquare(), "System isn't rectangular");
        assert(test.pseudoInverse.numCols() == test.matrix.numRows() && test.pseudoInverse.numRows() == test.matrix.numCols(), "Wrong dimensions of inverse");
        assert(Matrix.near(Matrix.mul(Matrix.mul(test.matrix, test.pseudoInverse), test.matrix), test.matrix), "Incorrect inverse");
        assert(test.leastSquaresSolution.size() == test.matrix.numCols(), "Inconsistent solution size");
        assert(test.rhs.size() == test.matrix.numRows(), "Inconsistent rhs size");
        assert(Vector.near(test.leastSquaresSolution, Matrix.postMulVec(test.pseudoInverse, test.rhs)), "Incorrect solution");
    }
})();

describe.skip('Linear solvers (dense rectangular matrices)', () => {
    test.each(rectSystemTestCases)("QR", (testData: RectSystemTestCase) => {
        for (const method of [linSolvers.ZeroingMethod.Givens, linSolvers.ZeroingMethod.Housholder]) {
            let solver = new linSolvers.QR(null, method, false);
            for (const makeCompact of [false, true]) {
                solver.makeCompact = makeCompact;
                const isColumn = testData.matrix.numRows() >= testData.matrix.numCols();
                const matrix = isColumn ? testData.matrix : testData.matrix.transpose();
                const inv = isColumn ? testData.pseudoInverse : testData.pseudoInverse.transpose();
                solver.factorize(matrix);
                expect(Matrix.lInfDistance(Matrix.mul(solver.Q, solver.R), matrix)).toBeLessThan(SmallTolerance);
                expect(Matrix.lInfDistance(solver.inverse(), inv)).toBeLessThan(SmallTolerance);
                expect(solver.R.isTriangular(true)).toBeTruthy();
                expect(solver.Q.isOrthogonal()).toBeTruthy();

                const transpose = matrix.transpose();
                solver.factorize(transpose);

                expect(Matrix.lInfDistance(Matrix.mul(solver.Q, solver.R), matrix)).toBeLessThan(SmallTolerance);
                expect(Matrix.lInfDistance(solver.inverse(), inv.transpose())).toBeLessThan(SmallTolerance);
                expect(solver.R.isTriangular(true)).toBeTruthy();
                expect(solver.Q.isOrthogonal()).toBeTruthy();
            }
        }
    });
});

test.skip("QR tests", () => {
    interface TestDataQR {
        matrix: Matrix;
        Q: Matrix;
        R: Matrix;
    };
    let squareTests: TestDataQR[] = [];
    {
        const A = new Matrix([12, -51, 4, 6, 167, -68, -4, 24, -41], 3, 3);
        const Q = new Matrix([6 / 7, -69 / 175, 58 / 175, 3 / 7, 158 / 175, -6 / 175, -2 / 7, 6 / 35, 33 / 35], 3, 3);
        const R = new Matrix([14, 21, -14, 0, 175, -70, 0, 0, -35], 3, 3);
        assert(Matrix.near(A, Matrix.mul(Q, R)), "Incorrect test data for QR decomposition");
        assert(R.isTriangular(true), "R matrix is expected to be triangular");
        assert(Q.isOrthogonal(), "Q matrix is expected to be orthogonal");
        squareTests.push({ matrix: A, Q: Q, R: R });
    }
    interface TestDataRectQR {
        matrix: Matrix;
        Q: Matrix;
        R: Matrix;
        compactQ: Matrix;
        compactR: Matrix;
    };
    // rectangular decomposition
    let rectTests: TestDataRectQR[] = [];
    {
        const A = new Matrix([1.08, 1.11, 0.04, 2.16, 3.03, 1.01, 2.16, 0.06, 0.04, 0, 1.08, 0.04], 4, 3);
        const Q = new Matrix([0.3, 0.2, -0.7, -0.4, 0.6, 0.7, 0.4, 0, 0.6, -0.4, -0.1, -0.5, 0, 0.4, -0.6, 0.4, 0.6, -0.4, -0.1, 0.7], 4, 4);
        const R = new Matrix([3.6, 1.9, 0.6, 2.2, 1.1, 0, 2.7, 0.7, -0.5, 1.4, 0, 0, 0.4, 0.3, 0.9, 0, 0, 0, 0.9, 0], 4, 5);
        const compactQ = Q.clone();
        compactQ.shrinkCols(A.numCols());
        const compactR = R.clone();
        compactR.shrinkRows(A.numCols());
        assert(Matrix.near(A, Matrix.mul(Q, R)), "Incorrect test data for QR decomposition");
        assert(Matrix.near(A, Matrix.mul(compactQ, compactR)), "Incorrect compact test data for QR decomposition");
        assert(compactR.isTriangular(true), "R matrix is expected to be triangular");
        assert(Q.isOrthogonal(), "Q matrix is expected to be orthogonal");
        assert(compactQ.isOrthogonal(), "compact Q matrix is expected to be orthogonal");
        rectTests.push({ matrix: A, Q: Q, R: R, compactQ: compactQ, compactR: compactR });
    }
    // rect decomposition
    for (const testData of rectTests) {
        for (const method of [linSolvers.ZeroingMethod.Givens, linSolvers.ZeroingMethod.Housholder]) {
            let solver = new linSolvers.QR(null, method, false);
            for (const makeCompact of [false, true]) {
                solver.makeCompact = makeCompact;
                solver.factorize(testData.matrix);
                expect(Matrix.lInfDistance(Matrix.mul(solver.Q, solver.R), testData.matrix)).toBeLessThan(SmallTolerance);
                expect(Matrix.lInfDistance(solver.R, testData.R)).toBeLessThan(SmallTolerance);
                expect(solver.R.isTriangular(true)).toBeTruthy();
                expect(solver.Q.isOrthogonal()).toBeTruthy();

                solver.factorize(testData.matrix.transpose());
                expect(Matrix.lInfDistance(Matrix.mul(solver.Q, solver.R), testData.matrix)).toBeLessThan(SmallTolerance);
                expect(Matrix.lInfDistance(solver.R, testData.R)).toBeLessThan(SmallTolerance);
                expect(solver.R.isTriangular(true)).toBeTruthy();
                expect(solver.Q.isOrthogonal()).toBeTruthy();
            }
        }
    }
});