import { SmallTolerance, assert } from "../../../../utils";

// Bisection + Newton
export default class HybridBisection {
    static solve(f: (x: number) => number, df: (x: number) => number, a: number, b: number, numIters: number, tolerance: number = SmallTolerance): number {
        let fa = f(a);
        if (Math.abs(fa) < tolerance) return a;
        let fb = f(b);
        if (Math.abs(fb) < tolerance) return b;
        assert(fb * fa <= 0, "Bad initial conditions");
        for (let i = 0; i < numIters; ++i) {
            let x = (a + b) * 0.5;
            let fx = f(x);
            let dfx = df(x);
            // todo (NI): figure out what to do with tolerance
            if (Math.abs(dfx) > tolerance) {
                let xn = x - fx / dfx;
                if (xn > a + tolerance && xn < b - tolerance)
                    x = xn;
                fx = f(x);
            }
            if (fx * fa < fx * fb) {
                b = x;
                fb = fx;
            } else {
                a = x;
                fa = fx;
            }
        }
        return Math.abs(fa) < Math.abs(fb) ? a : b;
    }
}