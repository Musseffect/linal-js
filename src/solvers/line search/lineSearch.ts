import Matrix from "../../dense/denseMatrix";
import { SmallestTolerance, Tolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import { ScalarFunction } from "../common/functions";

export abstract class LineSearchProblem {
    public abstract f(x: Vector): number;
    public abstract grad(x: Vector): Vector;
    public abstract hessian(x: Vector): Matrix;
}

export abstract class LineSearchScalarFunctionWrapper {
    function: ScalarFunction;
    public f(x: Vector): number {
        return this.function.f(x.get(0));
    }
    public grad(x: Vector): Vector {
        return new Vector([this.function.dfdx(x.get(0))]);
    }
    public hessian(x: Vector): Matrix {
        return new Matrix([this.function.dfddx(x.get(0))], 1, 1);
    }
}

export abstract class LineSearch {
    protected problem: LineSearchProblem;
    constructor(problem: LineSearchProblem) {
        this.problem = problem;
    }
    public abstract step(x: Vector, direction: Vector, initialStep: number): number;
}

export abstract class BacktrackingLineSearch extends LineSearch {
    /** min. decrease step */
    public minDecrease: number = Tolerance;
    /** coefficient for step length decrease */
    public tau: number = 0.5;
    /** max number of iterations */
    public maxNumIters: number = 20;
    public gradCoeff = 1e-4;
    protected abstract decreaseCondition(initialValue: number, x: Vector, direction: Vector, step: number, cosAngle: number): boolean;
    public override step(x: Vector, direction: Vector, initialStep: number): number {
        let initialValue = this.problem.f(x);
        let step = initialStep;
        const cosAngle = Vector.dot(this.problem.grad(x), direction);
        if (cosAngle > -SmallestTolerance) return this.minDecrease;
        const t = -this.gradCoeff * cosAngle;
        for (let iter = 0; iter < this.maxNumIters && step > this.minDecrease; ++iter) {
            if (this.decreaseCondition(initialValue, x, direction, step, cosAngle)) break;
            step = this.tau * step;
        }
        return step;
    }
    setTau(value: number) {
        assert(value > 0 && value < 1.0, "Invalid value for parameter tau");
        this.tau = value;
    }
    setGradCoeff(value: number) {
        assert(value > 0 && value < 1.0, "Invalid value for parameter c");
        this.gradCoeff = value;
    }
}