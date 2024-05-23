import Matrix from "../../dense/denseMatrix";
import { assert } from "../../utils";
import Vector from "../../dense/vector";

interface FourierCoeffs {
    a0: number;
    a: number[];
    b: number[];
}

class Fourier {
    static calcSeries(coeffs: FourierCoeffs, period: number, param: number) {
        assert(coeffs.a.length == coeffs.b.length, "Incorrect format");
        let value = coeffs.a0;
        for (let i = 0; i < coeffs.a.length; i++) {
            let angle = 2 * Math.PI * (i + 1) * param / period;
            value += coeffs.a[i] * Math.cos(angle) + coeffs.b[i] * Math.sin(angle);
        }
        return value;
    }
    static run(x: number[], y: number[], numHarmonics: number, period: number): FourierCoeffs {
        let m = Matrix.empty(numHarmonics * 2 + 1, numHarmonics * 2 + 1);
        let f = Vector.empty(numHarmonics * 2 + 1);
        for (let i = 0; i < x.length; i++) {
            f.set(f.get(0) + y[i], 0);
            for (let j = 0; j < numHarmonics; j++) {
                let angle = 2 * Math.PI * (j + 1) * x[i] / period;
                let ca = Math.cos(angle);
                let sa = Math.sin(angle);
                let a = f.get(1 + j) + ca * y[i];
                f.set(a, 1 + j);
                let b = f.get(1 + j + numHarmonics) + sa * y[i];
                f.set(b, 1 + j + numHarmonics);
                m.set(x.length, 0, 0);
                m.set(m.get(j + 1, 0) + ca, j + 1, 0);// first column
                m.set(m.get(j + numHarmonics + 1, 0) + sa, j + numHarmonics + 1, 0);// first column
                m.set(m.get(0, j + 1) + ca, 0, j + 1);//f irst row
                m.set(m.get(0, j + numHarmonics + 1) + sa, 0, j + numHarmonics + 1);// first row
                for (let k = 0; k < numHarmonics; k++) {
                    let angle2 = 2 * Math.PI * (k + 1) * x[i] / period;
                    let ca2 = Math.cos(angle2);
                    let sa2 = Math.sin(angle2);
                    m.set(m.get(j + 1, k + 1) + ca2 * ca, j + 1, k + 1);
                    m.set(m.get(j + 1, k + numHarmonics + 1) + sa2 * ca, j + 1, k + numHarmonics + 1);
                    m.set(m.get(j + numHarmonics + 1, k + 1) + ca2 * sa, j + numHarmonics + 1, k + 1);
                    m.set(m.get(j + numHarmonics + 1, k + numHarmonics + 1) + sa2 * sa, j + numHarmonics + 1, k + numHarmonics + 1);
                }
            }
        }
        let ab = Matrix.solve(m, f);
        return {
            a0: ab.data[0],
            a: ab.data.slice(1, 1 + numHarmonics),
            b: ab.data.slice(1 + numHarmonics)
        };
    }
}

export { Fourier, FourierCoeffs };