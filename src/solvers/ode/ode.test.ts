
import Vector from "../../dense/vector";
import * as ode from "./odeExports";


test("ODE", () => {
    class expOde implements ode.eode {
        param: number;
        constructor(param: number) {
            this.param = param;
        }
        f(x: Vector, t: number): Vector {
            return new Vector([this.param * x.get(0)]);
        }
        size(): number {
            return 1;
        }
    };
    const param = -1.0;
    let odeSystem = new expOde(param);
    let euler = new ode.euler(odeSystem, 0.005);
    let rk4 = new ode.rk4(odeSystem, 0.01);
    const t1 = 1.0;
    const x0 = new Vector([1.0]);
    let solution = Math.exp(param * t1) * x0.get(0);
    expect(rk4.solve(x0, 0, t1).get(0)).toBeCloseTo(solution);
    expect(euler.solve(x0, 0, t1).get(0)).toBeCloseTo(solution);
});