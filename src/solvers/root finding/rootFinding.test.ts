import { Tolerance, SmallTolerance, SmallestTolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import * as RootFinding from "./nonlinear systems/exports";


describe("Root finding: nonlinear", () => {
    test("Multidimensional", () => {
        let func = (p: Vector) => {
            const x = p.get(0);
            const y = p.get(1);
            return new Vector([20 * Math.log(x - y) - x - y - 6, 20 * Math.sin(0.7 * (x - y)) + 7 * (x + y)]);
        };
        let p0 = new Vector([0, -1]);
        let expectedRoot = new Vector([-0.46584782, -1.67846886]);
        let params = new RootFinding.NewtonRaphson.Params();
        params.fTolAbs = SmallTolerance;
        params.fDotTolAbs = SmallTolerance;
        params.jacobianEpsilon = SmallestTolerance;
        let root: Vector;
        expect(root = RootFinding.NewtonRaphson.Solver.solve(func, p0, 10, params)).not.toThrow();
        expect(func(expectedRoot).lInfNorm()).toBeLessThan(params.fTolAbs);
        expect(Vector.sub(expectedRoot, root)).toBeLessThan(Tolerance);
    })
});

test("1D root finding", () => {
    const func = (x: number): number => { return x * x * x - 3.0 * x * x * Math.cos(x * 4) + Math.sin(3.3 * x); };
    const df = (x: number): number => { return 3.0 * x * x - 6 * x * Math.cos(x * 4) + 12 * x * x * Math.sin(x * 4) + 3.3 * Math.cos(3.3 * x) };
    expect(func(Tolerance) - func(-Tolerance)).toBeCloseTo(df(0) * 2.0 * Tolerance);
    let xa = -0.5;
    let xb = 1.0;
    let solution = 0.0;
    expect(RootFinding.Bisection.solve(func, xa, xb, 20)).toBeCloseTo(solution);
    expect(RootFinding.RegulaFalsi.solve(func, xa, xb, 20)).toBeCloseTo(solution);
    expect(RootFinding.HybridBisection.solve(func, df, xa, xb, 20, Tolerance)).toBeCloseTo(solution);
});
