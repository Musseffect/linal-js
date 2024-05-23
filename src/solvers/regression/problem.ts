import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";

export abstract class LeastSquaresFunction {
    abstract numParameters(): number;
    abstract numVariables(): number;
    abstract f(x: Vector, params: Vector): number;
    abstract dfdp(x: Vector, p: Vector): Vector;
    abstract ddfdpdp(x: Vector, p: Vector): Matrix;
}

export class LeastSquaresSamples {
    arguments: Vector[];
    values: number[];
}

export abstract class LeastSquaresResiduals {
    public abstract numResiduals(): number;
    public abstract numParams(): number;
    public abstract f(p: Vector, idx: number): number;
    public abstract dfdp(p: Vector, idx: number): Vector;
    public abstract ddfdpdp(p: Vector, idx: number): Matrix;
    public error(p: Vector): number {
        let error = 0;
        for (let i = 0; i < this.numResiduals(); i++)
            error += Math.pow(this.f(p, i), 2.0);
        return error;
    }
}

export class LeastSquaresProblem extends LeastSquaresResiduals {
    samples: LeastSquaresSamples;
    func: LeastSquaresFunction;
    constructor(samples: LeastSquaresSamples, func: LeastSquaresFunction) {
        super();
        this.samples = samples;
        this.func = func;
    }
    public numResiduals(): number {
        return this.samples.arguments.length;
    }
    public numParams(): number {
        return this.func.numParameters();
    }
    public f(p: Vector, idx: number): number {
        return this.func.f(this.samples.arguments[idx], p) - this.samples.values[idx];
    }
    public dfdp(p: Vector, idx: number): Vector {
        return this.func.dfdp(this.samples.arguments[idx], p);
    }
    public ddfdpdp(p: Vector, idx: number): Matrix {
        return this.func.ddfdpdp(this.samples.arguments[idx], p);
    }
}