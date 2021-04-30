import { Network } from "https://cdn.jsdelivr.net/gh/umcconnell/leaf-detective@master/lib/index.min.js";
import { softplus } from "https://cdn.jsdelivr.net/gh/umcconnell/leaf-detective@master/lib/activations.min.js";
import { shuffle } from "https://cdn.jsdelivr.net/gh/umcconnell/leaf-detective@master/lib/helpers.min.js";

let white = document.getElementById("white-text");
let black = document.getElementById("black-text");
const NORMALIZER = 255;

let ranInt = (min, max) => Math.floor(Math.random() * (max + 1 - min)) + min;

let randomRGB = () => new Array(3).fill(0).map((_) => ranInt(0, 255));
let normalizeRGB = (color) => color.map((c) => c / NORMALIZER);
let rgb = (colors) => `rgb(${colors.slice(0, 3).join(",")})`;

function displayColor(color) {
    white.style.backgroundColor = rgb(color);
    black.style.backgroundColor = rgb(color);
}

function displayPrediction(predicted) {
    [white, black].forEach((box) => box.classList.remove("predicted"));
    (predicted == 0 ? black : white).classList.add("predicted");
}

function picker() {
    return new Promise((resolve) => {
        let listener = (ev) => {
            white.removeEventListener("click", listener);
            black.removeEventListener("click", listener);

            if (
                [...ev.target.classList].includes("white") ||
                [...ev.target.parentElement.classList].includes("white")
            ) {
                resolve(1);
            } else resolve(0);
        };

        white.addEventListener("click", listener);
        black.addEventListener("click", listener);
    });
}

function pick() {
    let currentColor = randomRGB();
    displayColor(currentColor);

    return [currentColor, picker];
}

async function quiz(times = 10, callback = () => "") {
    let choices = new Array(times).fill(0).map((_) => pick);
    let results = [];

    for (let pick of choices) {
        let [color, userChoice] = pick();
        callback(color);

        let label = await userChoice();
        results.push([color, label]);
    }

    return results;
}

async function main() {
    let ai = new Network([3, 3, 1], softplus)
        .connect()
        .addWeights()
        .addBiases();

    let asked = 0;
    let results = [];

    while (true) {
        let userChoices = await quiz(7, (color) => {
            asked++;
            if (asked > 7) {
                let prediction = ai.predict(normalizeRGB(color))[0];
                displayPrediction(Math.round(prediction));
            }
        });

        results.push(...userChoices);
        console.log("Training...");
        for (let i = 0; i < 500; i++) {
            shuffle(results).forEach(([color, label]) => {
                ai.feed(normalizeRGB(color))
                    .run()
                    .backpropagate([label], 0.1, 1);
            });
        }
        console.log("Done Training");
    }
}

main();
