import { SmallTolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import { ConvergenseFailureException } from "../linear systems/exceptions";
import LDLT from "../linear systems/ldlt";
import PartialPivLU from "../linear systems/partialPivLU";
import { Bounds, OptimizationProblem } from "./optimizationProblem";
import { LineSearchAlgorithm, initializeLineSearch } from "./utils";
import { PermutationMatrix, PermutationType } from "../../permutationMatrix";
import Matrix from "../../dense/denseMatrix";
import { TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import { SymmetricEigendecomposition } from "../linear systems/eigenvalues";
import LLT from "../linear systems/llt";


export enum NewtonRegularizationType {
    None,
    ClampNegativeEigenvalues,
    TikhonovIterative,
    TruncatedLLT
}
export class TruncatedLLT {
    private p: PermutationMatrix = null;
    private llt: Matrix = null;
    public get P() {
        return this.p;
    }
    public get L() {
        return new TriMatrixView(this.llt, TriMatrixType.lower);
    }
    public get LT() {
        return new TriMatrixView(this.llt, TriMatrixType.upper);
    }
    public get LLT() {
        return this.llt;
    }
    public get PLLTPT() {
        let result = Matrix.empty(this.llt.numRows(), this.llt.numCols());
        for (let i = 0; i < this.llt.numRows(); ++i) {
            for (let j = i; j < this.llt.numCols(); ++j) {
                let value = 0;
                for (let k = 0; k <= i; ++k)
                    value += this.llt.get(i, k) * this.llt.get(j, k);
                result.set(i, j, value);
                result.set(j, i, value);
            }
        }
        // let result = Matrix.mul(this.L.toMatrix(), this.LT.toMatrix());
        this.p.permuteInplace(result, PermutationType.Col);
        this.p.permuteInplace(result, PermutationType.Row);
        return result;
    }
    constructor(A?: Matrix, errTol: number = SmallTolerance) {
        if (A == undefined)
            return;
        this.factorize(A, errTol);
    }
    // http://www.dfg-spp1324.de/download/preprints/preprint076.pdf
    factorize(A: Matrix, errTol: number = SmallTolerance) {
        assert(A.isSquare(), "Matrix should be square");
        let it = 0;
        let llt = A.clone();
        let p = PermutationMatrix.identity(A.numRows(), PermutationType.Row);
        let error = llt.trace();
        while (error > errTol && it < A.numRows()) {
            let maxDiagIdx = it;
            let pivot = llt.get(it, it);
            for (let i = it + 1; i < A.numRows(); ++i) {
                let diagValue = llt.get(i, i);
                if (diagValue > pivot) {
                    pivot = diagValue;
                    maxDiagIdx = i;
                }
            }
            p.swap(it, maxDiagIdx);
            llt.swapRows(it, maxDiagIdx);
            llt.swapColumns(it, maxDiagIdx);
            pivot = Math.sqrt(pivot);
            llt.set(it, it, pivot);
            error = 0;
            for (let i = it + 1; i < A.numRows(); ++i) {
                let value = 0;
                for (let k = 0; k < it; ++k)
                    value += llt.get(k, it) * llt.get(k, i);
                let element = (llt.get(it, i) - value) / pivot;
                llt.set(it, i, element);
                llt.set(i, it, element);
                let diag = llt.get(i, i);
                diag -= element * element;
                llt.set(i, i, diag);
                error += diag;
            }
            it++;
        }
        while (it < A.numRows()) {
            llt.set(it, it, errTol);
            ++it;
        }
        this.p = p;
        this.llt = llt;
    }
    solve(rhs: Vector): Vector {
        assert(rhs.size() == this.llt.numRows(), "Incompatible RHS");
        let result = rhs.clone();
        this.p.permuteInplace(result);
        for (let row = 0; row < this.llt.numRows(); ++row) {
            let value = result.get(row);
            for (let col = 0; col < row; ++col)
                value -= this.llt.get(row, col) * result.get(col);
            value /= this.llt.get(row, row);
            result.set(row, value);
        }
        for (let row = this.llt.numRows() - 1; row >= 0; --row) {
            let value = result.get(row);
            for (let col = row + 1; col < this.llt.numRows(); ++col)
                value -= this.llt.get(row, col) * result.get(col);
            value /= this.llt.get(row, row);
            result.set(row, value);
        }
        this.p.permuteInplace(result);
        return result;
    }
}

// file:///C:/dev/lit/Math/Numerical%20methods/Numerical-Method-for-uncostrained-optimization-J.-E.-Dennis-Robert-B.-Schnabel.pdf page 337
export class Newton {
    lineSearchAlgo: LineSearchAlgorithm;
    initialStep: number;
    iterations: number;
    gradTolerance: number = SmallTolerance;
    dxAbsTol: number = SmallTolerance;
    solverTolerance: number = SmallTolerance;
    solverIterations: number;
    regularization: NewtonRegularizationType = NewtonRegularizationType.TikhonovIterative;

    constructor() {

    }
    private calcDirection(hessian: Matrix, grad: Vector): Vector {
        grad = Vector.negate(grad);
        if (this.regularization == NewtonRegularizationType.None) {
            let solver = new LDLT(hessian, this.solverTolerance);
            if (solver.L != null)
                return solver.solve(grad) as Vector;
            return grad;
        }
        for (let i = 0; i < hessian.numRows(); ++i) {
            for (let j = i + 1; j < hessian.numCols(); ++j) {
                let value = (hessian.get(i, j) + hessian.get(j, i)) / 2;
                hessian.set(i, j, value);
                hessian.set(j, i, value);
            }
        }
        switch (this.regularization) {
            case NewtonRegularizationType.TruncatedLLT:
                let solver = new TruncatedLLT(hessian, this.solverTolerance);
                return solver.solve(grad);
            case NewtonRegularizationType.ClampNegativeEigenvalues:
                let decomposition = new SymmetricEigendecomposition();
                decomposition.factorize(hessian, this.solverIterations);
                if (decomposition.D != null) {
                    let D = decomposition.D.clone();
                    for (let i = 0; i < D.size(); ++i)
                        D.set(i, Math.max(this.solverTolerance, D.get(i)));
                    return Matrix.postMulVec(decomposition.Q.transpose(), Vector.div(Matrix.postMulVec(decomposition.Q, grad), D));
                }
                break;
            case NewtonRegularizationType.TikhonovIterative:
                let lambda = Math.max(-Math.min.apply(null, hessian.diag().data), 0) + this.solverTolerance;
                let llt = new LLT();
                // todo: make this more sofisticated
                for (let i = 0; i < this.solverIterations; ++i) {
                    llt.factorize(Matrix.add(hessian, Matrix.fromDiagonal(Vector.generate(hessian.numRows(), () => lambda))));
                    if (llt.llt != null)
                        return llt.solveInplace(grad) as Vector;
                    lambda *= 2;
                }
                // fallback to gradient descent
                break;
        }
        return grad;
        // todo: test
    }
    solve(op: OptimizationProblem, x0: Vector, bounds?: Bounds): Vector {
        const lineSearch = initializeLineSearch(this.lineSearchAlgo, op);
        let x = x0.clone();
        let iter = 0;
        while (true) {
            let grad = op.dfdx(x);
            if (grad.l2Norm() < this.gradTolerance) return x;
            if (++iter == this.iterations) break;
            let hessian = op.dfdxdy(x);
            let direction = this.calcDirection(hessian, grad);
            let step = this.initialStep;
            if (bounds !== undefined)
                step = Math.min(step, bounds.intersect(x, direction));
            step = lineSearch.step(x, direction, step);
            x.addSelf(direction.scaleSelf(step));
        }
        throw new ConvergenseFailureException("Newton");
    }
    // calc H, calc QLQ decomposition and clamp negative eigenvalues or try some other scheme
}