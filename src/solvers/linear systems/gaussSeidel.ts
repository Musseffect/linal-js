import Matrix from "../../dense/denseMatrix";
import { assert, SmallTolerance, Tolerance } from "../../utils";
import Vector from "../../dense/vector";
import { ConvergenseFailureException } from "./exceptions";
import { PermutationMatrix, PermutationType } from "../../permutationMatrix";

const SolverName = "'GaussSeidel'";

export enum Status {
    NotExecuted,
    Success,
    IterationsExceeded
}

export default class GaussSeidel {
    private _status: Status = Status.NotExecuted;
    tolerance: number = SmallTolerance;
    maxIterations: number = 20;
    initialGuess?: Vector = null;
    reorderRows: boolean = false;
    throwOnFailure: boolean = true;
    public get status() {
        return this._status;
    }
    // todo: test
    public solve(m: Matrix, rhs: Vector): Vector {
        assert(m.width() == m.height(), "Matrix isn't square");
        assert(m.width() == rhs.size(), "Dimensions don't match");
        const rank = rhs.size();
        let result: Vector;
        if (this.initialGuess) {
            assert(rank == this.initialGuess.size(), "Initial guess doesn't match system rank");
            result = this.initialGuess.clone();
        } else {
            result = Vector.empty(rank);
        }
        let permutation: PermutationMatrix = null;
        if (this.reorderRows) {
            permutation = PermutationMatrix.identity(m.numRows(), PermutationType.Row);
            for (let step = 0; step < m.numRows(); ++step) {
                let pivotСost = -Infinity;
                let pivotRow = step;
                for (let row = step; row < m.numRows(); ++row) {
                    let cost = Math.abs(m.get(permutation.at(row), step)) * 2;
                    for (let col = 0; col < m.numCols(); ++col)
                        cost -= Math.abs(m.get(permutation.at(row), col));
                    if (cost > pivotСost) {
                        pivotСost = cost;
                        pivotRow = row;
                    }
                }
                if (pivotRow != step) permutation.swap(step, pivotRow);
            }
        }
        for (let it = 0; it < this.maxIterations; ++it) {
            let rhsApprox = Vector.empty(rank);
            for (let i = 0; i < rank; ++i) {
                let row = permutation ? permutation.at(i) : i;
                let sum = 0.0;
                for (let j = 0; j < i; ++j)
                    sum += m.get(row, j) * result.get(j);
                for (let j = i + 1; j < rank; ++j)
                    sum += m.get(row, j) * result.get(j);
                result.set(i, (rhs.get(row) - sum) / m.get(row, i));
                for (let j = 0; j < rank; ++j)
                    rhsApprox.set(j, rhsApprox.get(j) + m.get(j, i) * result.get(i));
            }
            if (rhsApprox.subSelf(rhs).lInfNorm() < this.tolerance) {
                //console.log("Reorder converged with Error {}, {}", it, Vector.sub(rhsApprox, rhs).lInfNorm())
                this._status = Status.Success;
                return result;
            }
        }
        //console.log("Reorder didn't converged with Error {}, {}", Vector.sub(Matrix.postMulVec(m, rhs), rhs).lInfNorm())
        this._status = Status.IterationsExceeded;
        if (this.throwOnFailure)
            throw new ConvergenseFailureException(SolverName);
    }
    static solve(m: Matrix, rhs: Vector, maxIterations: number, tolerance: number = SmallTolerance, initialGuess?: Vector): Vector {
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
        console.log("NotReorder")
        for (let it = 0; it < maxIterations; ++it) {
            let rhsApprox = Vector.empty(rank);
            for (let i = 0; i < rank; ++i) {
                let sum = 0.0;
                for (let j = 0; j < i; ++j)
                    sum += m.get(i, j) * result.get(j);
                for (let j = i + 1; j < rank; ++j)
                    sum += m.get(i, j) * result.get(j);
                result.set(i, (rhs.get(i) - sum) / m.get(i, i));
                for (let j = 0; j < rank; ++j)
                    rhsApprox.set(j, rhsApprox.get(j) + m.get(j, i) * result.get(i));
            }
            if (rhsApprox.subSelf(rhs).lInfNorm() < tolerance) {
                console.log("Default converged with Error {}, {}", it, Vector.sub(rhsApprox, rhs).lInfNorm())
                return result;
            }
        }
        console.log("Default didn't converged with Error {}, {}", Vector.sub(Matrix.postMulVec(m, rhs), rhs).lInfNorm())
        throw new ConvergenseFailureException(SolverName);
    }
}