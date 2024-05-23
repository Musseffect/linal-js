import { SparseMatrixCSR, SparseMatrixRowIterator } from "../../../sparse/sparseMatrix";
import { assert, SmallTolerance } from "../../../utils";
import Vector from "../../../dense/vector";
import { ConvergenseFailureException } from "../exceptions";

const SolverName = "'Jacobi'";

export default class Jacobi {
    static solve(m: SparseMatrixCSR, rhs: Vector, maxIterations: number, tolerance: number = SmallTolerance, initialGuess?: Vector) {
        assert(m.width() == m.height(), "Matrix isn't square");
        assert(m.width() == rhs.size(), "Dimensions don't match");
        const rank = rhs.size();
        let result: Vector;
        if (initialGuess) {
            assert(rank == initialGuess.size(), "Initial guess doesn't match system rank");
            result = initialGuess.clone();
        } else {
            result = Vector.empty(rank);
        }
        for (let it = 0; it < maxIterations; ++it) {
            let xNew = Vector.empty(rank);
            for (let row = 0; row < rank; ++row) {
                let sum = 0.0;
                let diagonalValue = 0.0;
                let it = new SparseMatrixRowIterator(m, row);
                while (!it.isDone()) {
                    let { value, colIdx } = it.advance();
                    if (colIdx == row)
                        diagonalValue = value;
                    else
                        sum += value * result.get(colIdx);
                }
                xNew.set(row, (rhs.get(row) - sum) / diagonalValue);
            }
            result = xNew;
            let rhsApprox = SparseMatrixCSR.postMul(m, result);
            if (rhsApprox.subSelf(rhs).lInfNorm() < tolerance)
                return result;
        }
        throw new ConvergenseFailureException(SolverName);
    }
}