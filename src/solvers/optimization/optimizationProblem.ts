import Matrix from "../../dense/denseMatrix";
import { forwardDifference, secondOrderDifference } from "../../numericalDifferentiation";
import { SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";

export abstract class OptimizationProblem {
    initialPoint: Vector;
    derivativeDelta: number = SmallTolerance;
    constructor(initialPoint: Vector) {
        this.initialPoint = initialPoint.clone();
    }
    setDerivativeDelta(value: number): void {
        this.derivativeDelta = value;
    }
    getInitialPoint(): Vector {
        return this.initialPoint;
    }
    getDimensions(): number {
        return this.initialPoint.size();
    }
    abstract numVariables(): number;
    abstract f(p: Vector): number;
    dfdx(p: Vector): Vector {
        return forwardDifference((x: Vector) => { return this.f(x); }, p, this.derivativeDelta);
    }
    dfdxdy(p: Vector): Matrix {
        return secondOrderDifference((x: Vector) => { return this.f(x); }, p, this.derivativeDelta);
    }
}

export abstract class Bounds {
    /**
     * Return distance to first intersection
     * @param ro ray origin
     * @param rd ray direction
     */
    abstract intersect(ro: Vector, rd: Vector): number;
};