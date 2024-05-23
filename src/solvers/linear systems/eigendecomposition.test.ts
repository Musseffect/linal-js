import Matrix from "../../dense/denseMatrix";
import { complex } from "../../complex";
import { RealSchurDecomposition, SymmetricEigendecomposition } from "./eigenvalues";
import { generatePolynomialWithComplexRoots } from "../../polynomial";
import { SmallTolerance, SmallestTolerance, Tolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import vec2 from "../../dense/vec2";

interface TestCase {
    matrix: Matrix;
    eigenvalues: complex[];
}

let testCases: { general: TestCase[], symmetric: TestCase[] } = {
    general: [],
    symmetric: []
};

(function () {

    let eigenvalues = [complex.real(1), complex.real(2), complex.real(-1), complex.real(3)];
    testCases.general.push({ matrix: generatePolynomialWithComplexRoots(eigenvalues).companionMatrix(), eigenvalues });
    testCases.general.push({
        matrix: new Matrix([
            1, 6, 5, 7,
            1, 2, 6, 1,
            2, 5, 3, 2,
            8, 1, 3, 4
        ], 4, 4),
        eigenvalues: [complex.real(-5.1467112), complex.real(-3.06044853), complex.real(4.28721786), complex.real(13.91994187)]
    });
    testCases.general.push({
        matrix: new Matrix([
            0, -0.5, 0, 0,
            0, 0.25, -0.5, 0,
            0, -0.125, 0.25, -0.5,
            0, 0.0625, -0.125, 0.250], 4, 4),
        eigenvalues: [complex.real(0), complex.real(0), complex.real(0.095491502813), complex.real(0.654508497187)]
    });
    testCases.general.push({
        matrix: new Matrix([
            -0.5, 0, 0, 0,
            0.25, 0, -0.5, 0,
            -0.125, 0, 0.25, -0.5,
            0.0625, 0, -0.125, 0.250], 4, 4),
        eigenvalues: [complex.real(0), complex.real(0), complex.real(0.5), complex.real(-0.5)]
    });
    testCases.general.push({
        matrix: new Matrix([
            0, -0.5, 0, 0,
            -0.5, 0.25, 0, 0,
            0.25, -0.125, 0, -0.5,
            -0.125, 0.0625, 0, 0.250], 4, 4),
        eigenvalues: [complex.real(0), complex.real(0.25), complex.real(-0.390388203202), complex.real(0.640388203202)]
    });
    testCases.general.push({
        matrix: new Matrix([
            0, -0.5, 0, 0,
            0, 0.25, -0.5, 0,
            -0.5, -0.125, 0.25, 0,
            0.250, 0.0625, -0.125, 0], 4, 4),
        eigenvalues: [complex.real(0), complex.real(-0.377438833123), new complex(0.438719416562, 0.372430883310), new complex(0.438719416562, -0.372430883310)]
    });
    // cyclic permutation
    testCases.general.push({
        matrix: new Matrix([
            0, 0, 0, 1,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        ], 4, 4),
        eigenvalues: [complex.real(1), complex.real(-1), new complex(0, 1), new complex(0, -1)]
    });
    testCases.symmetric.push({
        matrix: new Matrix([
            0, 0, 0, 0, 0,
            0, 1, 0, 3, 2,
            0, 0, 0, 0, 0,
            0, 3, 0, 4, 0,
            0, 2, 0, 0, 1
        ], 5, 5),
        eigenvalues: [complex.real(0), complex.real(0), complex.real(-1.902730770553), complex.real(1.812085272847), complex.real(6.090645497706)]
    });
    testCases.symmetric.push({
        matrix: new Matrix([
            2, 4, 0, 5, 0,
            4, 1, 0, 3, 0,
            0, 0, 0, 0, 0,
            5, 3, 0, 4, 0,
            0, 0, 0, 0, 0
        ], 5, 5),
        eigenvalues: [complex.real(0), complex.real(0), complex.real(-2.904715080557), complex.real(-0.682841652620), complex.real(10.587556733177)]
    });
    testCases.symmetric.push({
        matrix: new Matrix([
            1, 0.8034606, 0.7004717, 0.6996925, 0.6671458, 0.4958710, 0.5433092, 0.6900178, 0.6376572, 0.6162418,
            0.8034606, 1, 0.8219716, 0.8260105, 0.6896919, 0.4793830, 0.5699094, 0.7572619, 0.6680316, 0.6667194,
            0.7004717, 0.8219716, 1, 0.9921730, 0.6429816, 0.4169146, 0.5442465, 0.6656362, 0.7031941, 0.6645946,
            0.6996925, 0.8260105, 0.9921730, 1, 0.6356657, 0.4127679, 0.5514112, 0.6729881, 0.6984569, 0.6616017,
            0.6671458, 0.6896919, 0.6429816, 0.6356657, 1, 0.4392104, 0.4842821, 0.6302510, 0.6349091, 0.4030298,
            0.4958710, 0.4793830, 0.4169146, 0.4127679, 0.4392104, 1, 0.4303702, 0.5205868, 0.3700049, 0.3608651,
            0.5433092, 0.5699094, 0.5442465, 0.5514112, 0.4842821, 0.4303702, 1, 0.6379209, 0.4380425, 0.4685874,
            0.6900178, 0.7572619, 0.6656362, 0.6729881, 0.6302510, 0.5205868, 0.6379209, 1, 0.5819932, 0.5501563,
            0.6376572, 0.6680316, 0.7031941, 0.6984569, 0.6349091, 0.3700049, 0.4380425, 0.5819932, 1, 0.5190931,
            0.6162418, 0.6667194, 0.6645946, 0.6616017, 0.4030298, 0.3608651, 0.4685874, 0.5501563, 0.5190931, 1], 10, 10),
        eigenvalues: [complex.real(6.502731320756), complex.real(0.824501071), complex.real(0.631580363), complex.real(0.550487235), complex.real(0.406027528), complex.real(0.366323063), complex.real(0.298375573), complex.real(0.265420504), complex.real(0.147004966), complex.real(0.00754832)]
    });
    for (const testCase of testCases.symmetric)
        testCases.general.push(testCase);

    for (const testCase of testCases.general) {
        assert(testCase.matrix.isSquare(), "Matrix should be square");
        assert(testCase.eigenvalues.length == testCase.matrix.numRows(), "Number of eigenvalues should match size of the matrix");
        for (const eigenvalue of testCase.eigenvalues) {
            if (Math.abs(eigenvalue.y) != 0) continue;
            let mat = testCase.matrix.clone();
            for (let i = 0; i < testCase.matrix.numRows(); ++i)
                mat.set(i, i, mat.get(i, i) - eigenvalue.x);
            assert(Math.abs(mat.determinant()) < Tolerance, "det(A - Lambda * I) should be close to zero");
        }
    }
    for (const testCase of testCases.symmetric) {
        assert(testCase.matrix.isSymmetric(), "Matrix must be symmetric");
    }
})();


describe("Eigendecomposition", () => {
    test.each(testCases.general)("Schur decomposition %#", (testCase: TestCase) => {
        let decomposition = new RealSchurDecomposition(null);
        decomposition.tolerance = Tolerance;
        // todo investigate convergese with symmetric matrices
        decomposition.factorize(testCase.matrix, 20);
        expect(decomposition.D).not.toBeNull();
        expect(decomposition.Q.isOrthogonal()).toBeTruthy();
        expect(decomposition.D.isHessenberg()).toBeTruthy();
        // console.log(decomposition.D.toString());
        expect(Matrix.lInfDistance(testCase.matrix,
            Matrix.mul(Matrix.mul(decomposition.Q, decomposition.D), decomposition.Q.transpose()))).toBeCloseTo(0);
        let realEigenvalues = decomposition.realEigenvalues();
        let complexEigenvalues = decomposition.eigenvalues();
        expect(realEigenvalues.length).toEqual(testCase.eigenvalues.length);
        expect(complexEigenvalues.length).toEqual(testCase.eigenvalues.length);
        const compareComplexFunc = (a: complex, b: complex) => { return a.x - b.x; };
        const compareFunc = (a: number, b: number) => { return a - b; };
        let solution = testCase.eigenvalues.slice();
        solution.sort(compareComplexFunc);
        complexEigenvalues.sort(compareComplexFunc);
        realEigenvalues.sort(compareFunc);
        // console.log(realEigenvalues.toString());
        // console.log(complexEigenvalues.toString());
        // console.log(solution.toString());
        for (let i = 0; i < solution.length; ++i) {
            //console.log(` ${realEigenvalues[i]}, ${solution[i].x}, ${complexEigenvalues[i].toString()}`)
            expect(Math.abs(realEigenvalues[i] - solution[i].x)).toBeLessThanOrEqual(Tolerance);
            expect(vec2.sub(complexEigenvalues[i], solution[i]).lInfNorm()).toBeLessThanOrEqual(Tolerance);
        }
    });
    test.each(testCases.symmetric)("Symmetric eigendecomposition %#", (testCase: TestCase) => {
        let decomposition = new SymmetricEigendecomposition(null);
        decomposition.tolerance = SmallTolerance;
        decomposition.factorize(testCase.matrix, 20);
        expect(decomposition.D).not.toBeNull();
        expect(decomposition.Q.isOrthogonal()).toBeTruthy();
        expect(Matrix.lInfDistance(testCase.matrix,
            Matrix.mul(Matrix.mul(decomposition.Q, Matrix.fromDiagonal(decomposition.D)), decomposition.Q.transpose()))).toBeCloseTo(0);
        let eigenvalues = decomposition.D.data.slice();
        expect(eigenvalues.length).toEqual(testCase.eigenvalues.length);
        let solution = testCase.eigenvalues.slice();
        const compareComplexFunc = (a: complex, b: complex) => { return a.x - b.x; };
        const compareFunc = (a: number, b: number) => { return a - b; };
        solution.sort(compareComplexFunc);
        eigenvalues.sort(compareFunc);
        for (let i = 0; i < solution.length; ++i) {
            expect(Math.abs(eigenvalues[i] - solution[i].x)).toBeLessThanOrEqual(Tolerance);
        }
    });
});