import { swap } from "../utils";
import RandomNumberGenerator from "./generator";
import JSGenerator from "./js";



export function randomNormalDistr(generator: RandomNumberGenerator = new JSGenerator()) {
    const e1 = generator.randomUnit();
    const e2 = generator.randomUnit();
    return Math.sqrt(-2 * Math.log(e1)) * Math.cos(2 * Math.PI * e2);
}

export function shuffle<T>(array: T[], generator: RandomNumberGenerator = new JSGenerator()) {
    let size = array.length;
    for (let i = 0; i < size; ++i) {
        let targetIdx = Math.min(size - 1, Math.floor(generator.random(i, size)));
        swap(array, i, targetIdx);
    }
}