import { assert, Tolerance, SmallTolerance, SmallestTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { BacktrackingLineSearch, LineSearch, LineSearchProblem } from "./lineSearch";

export class ArmijoBacktracking extends BacktrackingLineSearch {
    constructor(problem: LineSearchProblem) {
        super(problem);
    }
    protected decreaseCondition(initialValue: number, x: Vector, direction: Vector, step: number, cosAngle: number): boolean {
        const curValue = this.problem.f(Vector.add(x, Vector.scale(direction, step)));
        return (initialValue - curValue) >= -step * cosAngle * this.gradCoeff;
    }
}

export class ArmijoTwoWayBacktracing extends LineSearch {
    gradCoeff: number;
    largestStep: number;
    tolerance: number;
    maxNumIters: number;
    tau: number;
    setLargestStep(value: number) {
        this.largestStep = value;
    }
    public step(x: Vector, direction: Vector, initialStep: number = 1.0): number {
        let initialValue = this.problem.f(x);
        let step = initialStep;
        const m = Vector.dot(this.problem.grad(x), direction);
        if (m > -SmallestTolerance) return this.tolerance;
        const t = -this.gradCoeff * m;
        assert(t < 0, "Invalid direction");
        let iter = 0;
        for (; iter < this.maxNumIters && step <= this.largestStep; ++iter) {
            let curValue = this.problem.f(Vector.add(x, Vector.scale(direction, step)));
            if (initialValue - curValue < step * t) break;
            step = step / this.tau;
        }
        for (; iter < this.maxNumIters && step >= this.tolerance; ++iter) {
            let curValue = this.problem.f(Vector.add(x, Vector.scale(direction, step)));
            if (initialValue - curValue >= step * t) break;
            step = step * this.tau;
        }
        return step;
    }
}
