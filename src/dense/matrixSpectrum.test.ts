import { RealSchurDecomposition, SymmetricEigendecomposition } from "../solvers/linear systems/eigenvalues";
import { makeHessenbergInplace } from "../solvers/linear systems/hessenbergMatrix";
import Matrix from "./denseMatrix";
import MatrixSpectrum from "./matrixSpectrum";
import { TriMatrixType, TriMatrixView } from "./matrixView";



describe("Spectrum", () => {
    test("", () => {
        const t = (mat: Matrix) => {
            let m = new TriMatrixView(mat, TriMatrixType.lower).toMatrix();
            //console.log(m.toString());
            let n = Matrix.sub(Matrix.empty(mat.numRows(), m.numRows()), new TriMatrixView(mat, TriMatrixType.upper, 0).toMatrix());
            //console.log(n.toString());
            m = m.inverse();
            let mn = Matrix.mul(m, n);
            //console.log(mn.toString());
            //console.log(`M-1 N spectral radius ${MatrixSpectrum.spectralRadius(mn)}`);
            let solver = new RealSchurDecomposition(mn, 40);
            //console.log(solver.D.toString());
            //console.log(solver.realEigenvalues().toString());
            expect(solver.D).not.toBeNull();
        };
        let m1 = new Matrix([
            0.0200, 0.0100, 0.0000, 0.0000,
            1.0000, 2.0000, 1.0000, 0.0000,
            0.0000, 1.0000, 2.0000, 1.0000,
            0.0000, 0.0000, 100.0000, 200.0000
        ], 4, 4);
        t(m1);
        // [0.012, 0.72, 2.78, 200.5]
        if (false) {
            let solver = new RealSchurDecomposition(m1, 20);
            expect(solver.D).not.toBeNull();
            //console.log(solver.realEigenvalues().toString());
            let m2 = new Matrix([
                0, 1,
                1, 1
            ], 2, 2);
            solver = new RealSchurDecomposition(m2, 20);
            expect(solver.D).not.toBeNull();
            //console.log(solver.realEigenvalues().toString());
            t(m2);
            // [-0.61, 1.61]
            let m3 = new Matrix([
                2, 3,
                5, 7
            ], 2, 2);
            solver = new RealSchurDecomposition(m3, 20);
            expect(solver.D).not.toBeNull();
            //console.log(solver.realEigenvalues().toString());
            t(m3);
            // [-0.109, 9.1]
            let m4 = new Matrix([
                4.0000, 12.0000, -16.0000,
                12.0000, 37.0000, -43.0000,
                -16.0000, -43.0000, 98.0000
            ], 3, 3);
            solver = new RealSchurDecomposition(m4, 20);
            expect(solver.D).not.toBeNull();
            // [0.019, 15.5, 123.48]
            //console.log(solver.realEigenvalues().toString());
            t(m4);
        }
        /*
        test.each(testCases.general)("Schur decomposition %#", (testCase: TestCase) => {
            let decomposition = new RealSchurDecomposition(null);
            decomposition.tolerance = Tolerance;
            // todo investigate convergese with symmetric matrices
            decomposition.factorize(testCase.matrix, 20);
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
        test.each(testCases.symmetric)("Symmetric eigendecomposition", (testCase: TestCase) => {
            let decomposition = new SymmetricEigendecomposition(null);
            decomposition.tolerance = SmallTolerance;
            decomposition.factorize(testCase.matrix, 20);
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
        });*/
    });
});