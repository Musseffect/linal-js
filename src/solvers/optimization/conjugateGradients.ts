import { SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { ConvergenseFailureException } from "../linear systems/exceptions";
import { Bounds, OptimizationProblem } from "./optimizationProblem";
import { LineSearchAlgorithm, initializeLineSearch } from "./utils";


export class ConjugateGradients {
    lineSearchAlgo: LineSearchAlgorithm;
    initialStep: number;
    gradTolerance: number;
    numIters: number;
    constructor(numIters: number = 20, lineSearchAlgo: LineSearchAlgorithm = LineSearchAlgorithm.Wolf, gradTolerance: number = SmallTolerance) {
        this.lineSearchAlgo = lineSearchAlgo;
        this.gradTolerance = gradTolerance;
        this.numIters = numIters;
    }
    solve(f: OptimizationProblem, x0: Vector, bounds?: Bounds): Vector {
        const lineSearch = initializeLineSearch(this.lineSearchAlgo, f);
        let x = x0.clone();
        let grad: Vector = f.dfdx(x);
        if (grad.l2Norm() < this.gradTolerance) return x;
        let prevDirection = grad;
        for (let i = 0; i < this.numIters; i++) {
            grad.scaleSelf(-1);
            // Polak Ribiere beta
            let beta = Vector.dot(prevDirection, Vector.sub(grad, prevDirection)) / Vector.dot(prevDirection, prevDirection);
            let direction: Vector = grad.clone();
            if (beta > 0)
                direction.addSelf(Vector.scale(prevDirection, beta));
            prevDirection = direction;
            let step = this.initialStep;
            if (bounds)
                step = Math.min(step, bounds.intersect(x, direction));
            step = lineSearch.step(x, direction, step);
            x.addSelf(direction.scaleSelf(step));
            // update grad and check it's magnitude
            grad = f.dfdx(x);
            if (grad.l2Norm() < this.gradTolerance) return x;
        }
        throw new ConvergenseFailureException("Conjugate descent");
    }
}