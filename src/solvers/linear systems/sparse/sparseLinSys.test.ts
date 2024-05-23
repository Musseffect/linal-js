import AbstractMatrix from "../../../abstractMatrix";
import { SparseMatrixCSR } from "../../../sparse/sparseMatrix";
import Triplet from "../../../sparse/triplet";
import { SmallTolerance, SmallestTolerance, Tolerance, assert } from "../../../utils";
import Vector from "../../../dense/vector";
import gaussSeidel from "./gaussSeidel";
import jacobi from "./jacobi";
import { ConjugateGradients } from "./conjugateGradients";
import Matrix from "../../../dense/denseMatrix";
import PartialPivLU from "./lu";
import IncompleteLU from "./incompleteLU";
import IncompleteLL from "./incompleteLL";
import IncompletePLU from "./incompletePLU";

interface TestCase {
    m: SparseMatrixCSR;
    rhs: Vector;
    exactSolution: Vector;
    determinant: number;
    inverse: Matrix;
}

interface Tests {
    posDef: TestCase[],
    general: TestCase[]
};

const tests: Tests = { posDef: [], general: [] };

(function () {
    let triplets: Triplet[] = [];
    triplets.push({ row: 0, column: 0, value: 4 });
    triplets.push({ row: 0, column: 1, value: 1 });
    triplets.push({ row: 0, column: 2, value: 2 });
    triplets.push({ row: 0, column: 3, value: 0.5 });
    triplets.push({ row: 0, column: 4, value: 2 });
    triplets.push({ row: 1, column: 0, value: 1 });
    triplets.push({ row: 1, column: 1, value: 0.5 });
    triplets.push({ row: 2, column: 0, value: 2 });
    triplets.push({ row: 2, column: 2, value: 3 });
    triplets.push({ row: 3, column: 0, value: 0.5 });
    triplets.push({ row: 3, column: 3, value: 0.625 });
    triplets.push({ row: 4, column: 0, value: 2 });
    triplets.push({ row: 4, column: 4, value: 16 });
    tests.posDef.push({
        m: SparseMatrixCSR.fromTriplets(5, 5, triplets),
        rhs: new Vector([6, 25, -11, 15, 10]),
        exactSolution: new Vector([-2995, 6040, 1993, 2420, 375]),
        determinant: 0.25,
        inverse: new Matrix([
            60, -120, -40, -48, -7.5,
            -120, 242, 80, 96, 15,
            -40, 80, 27, 32, 5,
            - 48, 96, 32, 40, 6,
            -7.5, 15, 5, 6, 1
        ], 5, 5)
    });
    triplets = [];
    triplets.push({ row: 0, column: 0, value: 2 });
    triplets.push({ row: 0, column: 1, value: -1 });
    triplets.push({ row: 1, column: 0, value: -1 });
    triplets.push({ row: 1, column: 1, value: 2 });
    triplets.push({ row: 1, column: 2, value: -1 });
    triplets.push({ row: 2, column: 1, value: -1 });
    triplets.push({ row: 2, column: 2, value: 2 });
    triplets.push({ row: 2, column: 3, value: -1 });
    triplets.push({ row: 3, column: 2, value: -1 });
    triplets.push({ row: 3, column: 3, value: 2 });
    triplets.push({ row: 3, column: 4, value: -1 });
    triplets.push({ row: 4, column: 3, value: -1 });
    triplets.push({ row: 4, column: 4, value: 2 });
    triplets.push({ row: 4, column: 5, value: -1 });
    triplets.push({ row: 5, column: 4, value: -0.5 });
    triplets.push({ row: 5, column: 5, value: 1 });
    triplets.push({ row: 5, column: 6, value: -0.5 });
    triplets.push({ row: 6, column: 5, value: -1 });
    triplets.push({ row: 6, column: 6, value: 2 });
    triplets.push({ row: 6, column: 7, value: -1 });
    triplets.push({ row: 7, column: 6, value: -0.5 });
    triplets.push({ row: 7, column: 7, value: 1 });
    triplets.push({ row: 7, column: 8, value: -0.5 });
    triplets.push({ row: 8, column: 6, value: 0.5 });
    triplets.push({ row: 8, column: 7, value: -1.5 });
    triplets.push({ row: 8, column: 8, value: 4 });
    triplets.push({ row: 8, column: 9, value: -1.5 });
    triplets.push({ row: 9, column: 8, value: -1 });
    triplets.push({ row: 9, column: 9, value: 2 });
    tests.general.push({
        m: SparseMatrixCSR.fromTriplets(10, 10, triplets),
        rhs: new Vector([1, 3, 4, -4, 2, 3, 1, -2, -3, 1]),
        exactSolution: new Vector([
            514 / 83,
            945 / 83,
            1127 / 83,
            977 / 83,
            1159 / 83,
            1175 / 83,
            693 / 83,
            128 / 83,
            -105 / 83,
            -11 / 83
        ]),
        determinant: 10.375,
        inverse: new Matrix([
            74 / 83, 65 / 83, 56 / 83, 47 / 83, 38 / 83, 58 / 83, 20 / 83, 26 / 83, 4 / 83, 3 / 83,
            65 / 83, 130 / 83, 112 / 83, 94 / 83, 76 / 83, 116 / 83, 40 / 83, 52 / 83, 8 / 83, 6 / 83,
            56 / 83, 112 / 83, 168 / 83, 141 / 83, 114 / 83, 174 / 83, 60 / 83, 78 / 83, 12 / 83, 9 / 83,
            47 / 83, 94 / 83, 141 / 83, 188 / 83, 152 / 83, 232 / 83, 80 / 83, 104 / 83, 16 / 83, 12 / 83,
            38 / 83, 76 / 83, 114 / 83, 152 / 83, 190 / 83, 290 / 83, 100 / 83, 130 / 83, 20 / 83, 15 / 83,
            29 / 83, 58 / 83, 87 / 83, 116 / 83, 145 / 83, 348 / 83, 120 / 83, 156 / 83, 24 / 83, 18 / 83,
            20 / 83, 40 / 83, 60 / 83, 80 / 83, 100 / 83, 240 / 83, 140 / 83, 182 / 83, 28 / 83, 21 / 83,
            11 / 83, 22 / 83, 33 / 83, 44 / 83, 55 / 83, 132 / 83, 77 / 83, 208 / 83, 32 / 83, 24 / 83,
            2 / 83, 4 / 83, 6 / 83, 8 / 83, 10 / 83, 24 / 83, 14 / 83, 68 / 83, 36 / 83, 27 / 83,
            1 / 83, 2 / 83, 3 / 83, 4 / 83, 5 / 83, 12 / 83, 7 / 83, 34 / 83, 18 / 83, 55 / 83
        ], 10, 10)
    });
    const checkTest = (test: TestCase, testID: number) => {
        assert(test.m.isSquare(), `Expected square matrix ${testID}`);
        assert(test.exactSolution.size() == test.m.numCols(), `Inconsistent system size ${testID}`);
        assert(test.rhs.size() == test.m.numCols(), `Inconsistent system size ${testID}`);
        assert(Vector.near(test.rhs, SparseMatrixCSR.postMul(test.m, test.exactSolution), SmallestTolerance), `Incorrect solution ${testID}`);
        assert(Vector.near(test.exactSolution, Matrix.postMulVec(test.inverse, test.rhs), SmallestTolerance), `Incorrect inverse ${testID}`);
    };
    let testID = 0;
    for (const test of tests.posDef) {
        assert(test.m.isSymmetric(), "Expected symmetric matrix");
        tests.general.push(test);
    }
    testID = 0;
    for (const test of tests.general) {
        checkTest(test, testID++);
    }
})();

describe('Linear solvers (sparse square matrices)', () => {
    describe.each(tests.posDef)('Symmetric positive definite matrices %#', (testCase: TestCase) => {
        expect(testCase.m.isSymmetric()).toBeTruthy();
        describe('Iterative', () => {
            test('ConjGrad', () => {
                let result = ConjugateGradients.solve(testCase.m, testCase.rhs, 20);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(SmallTolerance);
            });
        });
    });
    describe.each(tests.general)('General matrices %#', (testCase: TestCase) => {
        describe('Factorizations', () => {
            test('PPLU', () => {
                let solver = new PartialPivLU();
                expect(() => solver.factorize(testCase.m)).not.toThrow();
                expect(solver.LU).not.toBeNull();
                expect(solver.LU.isValid()).toBeTruthy();
                expect(solver.P).not.toBeNull();
                expect(SparseMatrixCSR.lInfDistance(solver.P.inverse().permuteSparseMatrix(SparseMatrixCSR.mul(solver.L, solver.U)), testCase.m)).toBeCloseTo(0);
                expect(Matrix.lInfDistance(solver.inverse(), testCase.inverse)).toBeCloseTo(0);
                expect(Vector.lInfDistance(solver.solve(testCase.rhs), testCase.exactSolution)).toBeCloseTo(0);
                expect(solver.determinant()).toBeCloseTo(testCase.determinant, 4);
            });
        });
        describe.skip('Iterative', () => {
            const initialGuess = Vector.add(testCase.exactSolution, Vector.generate(testCase.exactSolution.size(), (i: number) => { return (Math.random() - 0.5) }));
            test('gauss-zeidel', () => {
                let result = gaussSeidel.solve(testCase.m, testCase.rhs, 120, SmallTolerance, initialGuess);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
            test('jacobi', () => {
                let result = jacobi.solve(testCase.m, testCase.rhs, 120, SmallTolerance, initialGuess);
                expect(Vector.lInfDistance(result, testCase.exactSolution)).toBeLessThanOrEqual(Tolerance);
            });
        });
    });
});

describe('Incomplete LU', () => {
    // Incomplete decompositions of dense matrix should calculate exact LU decomposition
    describe("Dense matrix", () => {
        const matrix = new Matrix([
            1, 2, 3,
            4, 4, 4,
            -1, -2, -6
        ], 3, 3);
        const LU = new Matrix([
            1, 2, 3,
            4, -4, -8,
            -1, 0, -3
        ], 3, 3);
        const rhs = new Vector([14, 24, -23]);
        const solution = new Vector([1, 2, 3]);
        const matrixCSR = SparseMatrixCSR.fromDense(matrix, 0);
        test("LU", () => {
            let solver = new IncompleteLU();
            solver.factorize(matrixCSR);
            expect(solver.LU).not.toBeNull();
            expect(solver.LU.isValid()).toBeTruthy();

            expect(SparseMatrixCSR.compareSparsity(matrixCSR, solver.LU)).toBeTruthy();
            expect(Matrix.lInfDistance(LU, solver.LU.toDense())).toBeCloseTo(0);
            expect(Vector.lInfDistance(solver.solve(rhs), solution)).toBeCloseTo(0);
        });
        test("PLU", () => {
            let solver = new IncompletePLU()
            solver.factorize(matrixCSR);
            expect(solver.LU).not.toBeNull();
            expect(solver.LU.isValid()).toBeTruthy();

            expect(SparseMatrixCSR.compareSparsity(matrixCSR, solver.LU)).toBeTruthy();
            expect(Matrix.lInfDistance(matrix, solver.P.inverse().permuteMatrix(SparseMatrixCSR.mul(solver.L, solver.U).toDense()))).toBeCloseTo(0);
            expect(Vector.lInfDistance(solver.solve(rhs), solution)).toBeCloseTo(0);
        });
    });

    interface TestCaseLU {
        matrix: Matrix,
        incompleteLU: Matrix,
        incompletePLU: Matrix,
        rhs: Vector,
        incompleteLUSolution: Vector,
        incompletePLUSolution: Vector,
        solution: Vector
    };
    let testData: TestCaseLU[] = [];
    testData.push({
        matrix: new Matrix([
            1, 0, 2, 0,
            -2, 4, 0, 2,
            0, 1, 2, 0,
            0, 3, 3, 3
        ], 4, 4),
        incompleteLU: new Matrix([
            1, 0, 2, 0,
            -2, 4, 0, 2,
            0, 0.25, 2, 0,
            0, 0.75, 1.5, 1.5
        ], 4, 4),
        incompletePLU: new Matrix([
            -0.5, 0, 2, 0,
            - 2, 4, 0, 2,
            0, 1 / 3, 0.5, 0,
            0, 3, 3, 3
        ], 4, 4),
        rhs: new Vector([7, 2, 10, 27]),
        incompleteLUSolution: new Vector([1, 2, 3, 4]),
        incompletePLUSolution: new Vector([12, 8, 4, -3]),
        solution: new Vector([-15, -12, 11, 10])
    });
    // todo: test this as preconditioner for gauss seidel


    test.each(testData)("LU %#", (testCase: TestCaseLU) => {
        let matrixCSR = SparseMatrixCSR.fromDense(testCase.matrix, 0);
        let solver = new IncompleteLU()
        solver.factorize(matrixCSR);
        expect(solver.LU).not.toBeNull();
        expect(solver.LU.isValid()).toBeTruthy();

        expect(SparseMatrixCSR.compareSparsity(matrixCSR, solver.LU)).toBeTruthy();
        expect(Matrix.lInfDistance(testCase.incompleteLU, solver.LU.toDense())).toBeCloseTo(0);
        expect(Vector.lInfDistance(solver.solve(testCase.rhs), testCase.incompleteLUSolution)).toBeCloseTo(0);
    });
    test.each(testData)("PLU %#", (testCase: TestCaseLU) => {
        let matrixCSR = SparseMatrixCSR.fromDense(testCase.matrix, 0);
        let solver = new IncompletePLU()
        solver.factorize(matrixCSR);
        expect(solver.LU).not.toBeNull();
        expect(solver.LU.isValid()).toBeTruthy();

        expect(SparseMatrixCSR.compareSparsity(matrixCSR, solver.P.inverse().permuteSparseMatrix(solver.LU))).toBeTruthy();
        expect(Matrix.lInfDistance(testCase.incompletePLU, solver.P.inverse().permuteSparseMatrix(solver.LU).toDense())).toBeCloseTo(0);
        expect(Vector.lInfDistance(solver.solve(testCase.rhs), testCase.incompletePLUSolution)).toBeCloseTo(0);
    });
    test.each(testData)("LU iterations %#", (testCase: TestCaseLU) => {
        let matrixCSR = SparseMatrixCSR.fromDense(testCase.matrix, 0);
        let solver = new IncompleteLU()
        solver.factorize(matrixCSR);
        expect(solver.LU).not.toBeNull();
        expect(solver.LU.isValid()).toBeTruthy();

        let x = solver.solve(testCase.rhs);
        for (let i = 0; i < 30; ++i) {
            let r = Vector.sub(testCase.rhs, SparseMatrixCSR.postMul(matrixCSR, x));
            let d = solver.solve(r);
            x.addSelf(d);
        }
        expect(Vector.lInfDistance(x, testCase.solution)).toBeCloseTo(0);
    });
    test.skip.each(testData)("PLU iterations %#", (testCase: TestCaseLU) => {
        let matrixCSR = SparseMatrixCSR.fromDense(testCase.matrix, 0);
        let solver = new IncompletePLU()
        solver.factorize(matrixCSR);
        expect(solver.LU).not.toBeNull();
        expect(solver.LU.isValid()).toBeTruthy();

        let x = solver.solve(testCase.rhs);
        for (let i = 0; i < 10; ++i) {
            let r = Vector.sub(testCase.rhs, SparseMatrixCSR.postMul(matrixCSR, x));
            let d = solver.solve(r);
            x.addSelf(d);
        }
        expect(Vector.lInfDistance(x, testCase.solution)).toBeCloseTo(0);
    });
});

describe('Incomplete LL', () => {
    interface TestCaseLL {
        matrix: Matrix;
        incompleteL: Matrix;
        rhs: Vector;
        incompleteSolution: Vector;
    }
    let testData: TestCaseLL[] = [];
    testData.push({
        matrix: new Matrix([
            5, -2, 0, -2, 2,
            -2, 5, -2, 0, 0,
            0, -2, 5, -2, 0,
            -2, 0, -2, 5, -2,
            - 2, 0, 0, -2, 5
        ], 5, 5),
        incompleteL: new Matrix([
            2.24, 0, 0, 0, 0,
            -0.89, 2.05, 0, 0, 0,
            0, -0.98, 2.01, 0, 0,
            -0.89, 0, -0.99, 1.79, 0,
            -0.89, 0, 0, -1.56, 1.33
        ], 5, 5),
        rhs: new Vector([-17, 9.2, 3, 3.6, 16.6]),
        incompleteSolution: new Vector([1, 2, 3, 4, 5])
    });
    testData.push({
        matrix: new Matrix([
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        ], 3, 3),
        incompleteL: new Matrix([
            2, 0, 0,
            6, 1, 0,
            -8, 5, 3
        ], 3, 3),
        rhs: new Vector([-20, -43, 192]),
        incompleteSolution: new Vector([1, 2, 3])
    });
    test.each(testData)("Test #%", (testCase: TestCaseLL) => {
        let matrixCSR = SparseMatrixCSR.fromDense(testCase.matrix, 0);
        let solver = new IncompleteLL(matrixCSR);
        expect(solver.L).not.toBeNull();
        expect(SparseMatrixCSR.compareSparsity(matrixCSR, SparseMatrixCSR.add(solver.L, solver.L.transpose()))).toBeTruthy();
        expect(SparseMatrixCSR.lInfDistance(solver.L, SparseMatrixCSR.fromDense(testCase.incompleteL, 0))).toBeCloseTo(0);
        expect(Vector.lInfDistance(solver.solve(testCase.rhs), testCase.incompleteSolution)).toBeCloseTo(0);
    });
});

/*
const singularTrivialMatrixTriplets: Triplet[] = [{ row: 1, column: 1, value: 1 },
    { row: 2, column: 2, value: 1 }, { row: 0, column: 0, value: 2 }, { row: 3, column: 3, value: 1 }, { row: 5, column: 5, value: 1 }];
    const singularSparseTrivialMatrix = SparseMatrixCSR.fromTriplets(6, 6, singularTrivialMatrixTriplets);
    const singularDenseTrivialMatrix = Matrix.fromTriplets(6, 6, singularTrivialMatrixTriplets);
    
    const singularNonTrivialMatrixTriplets: Triplet[] = [
        { row: 0, column: 0, value: 1 },
        { row: 0, column: 2, value: 3 },
        { row: 0, column: 4, value: 1 },
        { row: 1, column: 0, value: 2 },
        { row: 1, column: 1, value: 2 },
        { row: 1, column: 5, value: 2 },
        { row: 2, column: 0, value: 2 },
        { row: 2, column: 1, value: 2 },
        { row: 2, column: 2, value: 3 },
        { row: 2, column: 3, value: 1 },
        { row: 2, column: 5, value: 3 },
        { row: 3, column: 4, value: 4 },
        { row: 4, column: 0, value: 3 },
        { row: 4, column: 1, value: -3 },
        { row: 4, column: 2, value: -1 },
        { row: 4, column: 4, value: 2 },
        { row: 5, column: 1, value: 2 },
        { row: 5, column: 3, value: 2 },
        { row: 5, column: 5, value: 4 }
    ];
    const singularSparseNonTrivialMatrix = SparseMatrixCSR.fromTriplets(6, 6, singularNonTrivialMatrixTriplets);
    const singularDenseNonTrivialMatrix = Matrix.fromTriplets(6, 6, singularNonTrivialMatrixTriplets);
    
    const nonSingularMatrixTriplets: Triplet[] = [{ row: 0, column: 0, value: 1 }, { row: 0, column: 2, value: 3 }, { row: 0, column: 4, value: 1 },
    { row: 1, column: 0, value: 2 }, { row: 1, column: 1, value: 2 }, { row: 1, column: 5, value: 2 }, { row: 2, column: 1, value: 2 }, { row: 2, column: 3, value: 1 }
        , { row: 2, column: 5, value: 3 }, { row: 3, column: 4, value: 4 }, { row: 4, column: 0, value: 3 }, { row: 4, column: 1, value: -3 }, { row: 4, column: 2, value: -1 }
        , { row: 4, column: 4, value: 2 }, { row: 5, column: 1, value: 2 }, { row: 5, column: 3, value: 2 }, { row: 5, column: 5, value: 4 }];
    const nonSingularSparseMatrix = SparseMatrixCSR.fromTriplets(6, 6, nonSingularMatrixTriplets);
    const nonSingularDenseMatrix = Matrix.fromTriplets(6, 6, nonSingularMatrixTriplets);
    const nonSingularMatrixDeterminant = 144.0;
    */