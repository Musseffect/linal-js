import { RealSchurDecomposition, SymmetricEigendecomposition } from "../solvers/linear systems/eigenvalues";
import Matrix from "./denseMatrix";


export default class MatrixSpectrum {
    static conditionNumber(matrix: Matrix, numIters: number = 20): number {
        const calcCondition = (array: number[]) => {
            return array.reduce((a, b) => Math.max(a, Math.abs(b)), 0) / array.reduce((a, b) => Math.min(a, Math.abs(b)), Infinity);
        };
        if (matrix.isSymmetric()) {
            let solver = new SymmetricEigendecomposition(matrix, numIters);
            if (solver.D != null) return calcCondition(solver.D.data);
        }
        let solver = new RealSchurDecomposition(matrix, numIters);
        if (solver.D != null)
            return calcCondition(solver.realEigenvalues());
        return NaN;
    }
    static spectralRadius(matrix: Matrix, numIters: number = 20): number {
        const calcRadius = (array: number[]) => {
            return array.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
        };
        if (matrix.isSymmetric()) {
            let solver = new SymmetricEigendecomposition(matrix, numIters);
            if (solver.D != null) return calcRadius(solver.D.data);
        }
        let solver = new RealSchurDecomposition(matrix, numIters);
        if (solver.D != null)
            return calcRadius(solver.realEigenvalues());
        return NaN;
    }
}