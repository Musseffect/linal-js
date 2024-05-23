import Matrix from "../../../../dense/denseMatrix";
import { assert, assertFail, SmallTolerance } from "../../../../utils";
import Vector from "../../../../dense/vector";
import { LineSearch, LineSearchProblem } from "../../../line search/lineSearch";
import { ConvergenseFailureException } from "../../../linear systems/exceptions";
import FullPivLU from "../../../linear systems/fullPivLU";
import PartialPivLU from "../../../linear systems/partialPivLU";
import { createLineSearch, initializeLineSearch, LineSearchAlgorithm } from "../../../optimization/utils";


const SolverName = "Newton";
class Params {
    jacobian?: (x: Vector) => Matrix;
    fDotTolAbs: number = SmallTolerance;
    fTolAbs: number = SmallTolerance;
    jacobianEpsilon: number = SmallTolerance;
    lineSearchAlgo?: LineSearchAlgorithm;
    step: number;
}

class Solver {
    private static numericalJacobian(f: (x: Vector) => Vector, x: Vector, step: number): Matrix {
        let result: Matrix = Matrix.empty(x.size(), x.size());
        // forward difference
        let f0 = f(x);
        for (let i = 0; i < x.size(); ++i) {
            let scalar = x.get(i);
            x.set(i, scalar + step);
            let dfdx = f(x).subSelf(f0).scaleSelf(1.0 / step);
            x.set(i, scalar);
            result.setColumn(i, dfdx);
        }
        return result;
    }
    static solve(f: (x: Vector) => Vector, x0: Vector, numIters: number, params: Params = new Params()): Vector {
        let x = x0.clone();
        let calcJacobian = params.jacobian ? params.jacobian : (y: Vector) => { return Solver.numericalJacobian(f, y, SmallTolerance) };
        let lineSearchProblem: LineSearchProblem = {
            f: (x: Vector): number => {
                return f(x).l2Norm() / 2;
            },
            grad: (x: Vector): Vector => {
                return Matrix.postMulVec(calcJacobian(x), f(x));
            },
            hessian: (x: Vector): Matrix => {
                assertFail("shouldn't be called");
                return Matrix.empty(1, 1);
            }
        };
        let lineSearch: LineSearch | null = params.lineSearchAlgo ? createLineSearch(params.lineSearchAlgo, lineSearchProblem) : null;
        let fVec = Vector.negate(f(x));
        let maxNorm = fVec.lInfNorm();
        for (let iter = 0; iter < numIters; ++iter) {
            if (maxNorm < params.fTolAbs) return x;
            let J: Matrix = calcJacobian(x);
            if (J.lInfNorm() < params.fDotTolAbs)
                return x;
            try {
                // TODO: use QR with column pivoting for non square and
                // non-full rank systems
                let dx = PartialPivLU.solve(J, fVec);
                let step = params.step;
                if (lineSearch)
                    lineSearch.step(x, dx, step);
                x.addSelf(dx.scaleSelf(step));
            } catch (exc) {
                break;
            }
            fVec = Vector.negate(f(x));
            maxNorm = fVec.lInfNorm();
        }
        throw new ConvergenseFailureException(SolverName);
    }
    // todo: add line search, bounds, etc.
    // trust region methods
    // check rank
}

export default { Solver, Params };