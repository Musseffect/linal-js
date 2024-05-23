import Matrix from "../../dense/denseMatrix";
import { assert, SmallTolerance } from "../../utils";
import vector from "../../dense/vector";
import { ConvergenseFailureException } from "./exceptions";
// https://en.wikipedia.org/wiki/Successive_over-relaxation

const SolverName = "'SOR'";

/** Successive over-relaxation
 * 
 */
export default class SOR {
    static solve(m: Matrix, rhs: vector, maxIterations: number, weight: number, tolerance: number = SmallTolerance, initialGuess?: vector) {
        assert(m.width() == m.height(), "Matrix isn't square");
        assert(m.width() == rhs.size(), "Dimensions don't match");
        const rank = rhs.size();
        let result: vector;
        if (initialGuess) {
            assert(rank == initialGuess.size(), "Initial guess doesn't match system rank");
            result = initialGuess.clone();
        } else {
            result = vector.empty(rank);
        }
        for (let it = 0; it < maxIterations; ++it) {
            let rhsApprox = vector.empty(rank);
            for (let i = 0; i < rank; ++i) {
                let sum = 0.0;
                for (let j = 0; j < i; ++j)
                    sum += m.get(i, j) * result.get(j);
                for (let j = i + 1; j < rank; ++j)
                    sum += m.get(i, j) * result.get(j);
                result.set(i, (1 - weight) * result.get(i) + weight * (rhs.get(i) - sum) / m.get(i, i));
                for (let j = 0; j < rank; ++j)
                    rhsApprox.set(j, rhsApprox.get(j) + m.get(j, i) * result.get(i));
            }
            if (rhsApprox.subSelf(rhs).lInfNorm() < tolerance)
                return result;
        }
        throw new ConvergenseFailureException(SolverName);
    }
}