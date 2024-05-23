import { SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { LineSearch } from "../line search/lineSearch";
import { ConvergenseFailureException } from "../linear systems/exceptions";
import { Bounds, OptimizationProblem } from "./optimizationProblem";
import { LineSearchAlgorithm, initializeLineSearch } from "./utils";

export default class GradientDescent {
    lineSearchAlgo: LineSearchAlgorithm;
    gradTolerance: number;
    step: number;
    numIters: number;
    constructor(numIters: number = 20, lineSearchAlgo: LineSearchAlgorithm = LineSearchAlgorithm.Wolf, gradTolerance: number = SmallTolerance) {
        this.lineSearchAlgo = lineSearchAlgo;
        this.gradTolerance = gradTolerance;
        this.numIters = numIters;
    }
    solve(f: OptimizationProblem, x0: Vector, bounds?: Bounds): Vector {
        const lineSearch = initializeLineSearch(this.lineSearchAlgo, f);
        let x = x0.clone();
        let iter = 0;
        while (true) {
            let direction = Vector.negate(f.dfdx(x));
            if (direction.l2Norm() < this.gradTolerance) return x;
            if (++iter == this.numIters) break;
            let step = this.step;
            if (bounds)
                step = Math.min(step, bounds.intersect(x, direction));
            step = lineSearch.step(x, direction, step);
            x.addSelf(direction.scaleSelf(step));
        }
        throw new ConvergenseFailureException("Gradient descent");
    }
}