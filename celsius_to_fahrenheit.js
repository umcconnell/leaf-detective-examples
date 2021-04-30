let { Network } = require("../lib/index.js");
let { plus, shuffle } = require("../lib/helpers.js");
let { relu } = require("../lib/activations.js");

const NORMALIZER = 212;

console.log(`Celsius to Fahrenheit conversion:
################
The neural network consists of 2 hidden layers with the size 4 and 2.
It is given a temperature between 1 and 100 in celsius and expected to return
the corresponding temperature in degrees fahrenheit.
################
`);

let t0 = new Date().getTime();

// 1. Create network
let network = new Network([1, 4, 2, 1], relu)
    .connect()
    .addWeights()
    .addBiases();

// 2. Generate Data
let data = new Array(600).fill(0).map(_ => {
    let celsius = Math.floor(Math.random() * (100 - 0 + 1)) + 0;

    //  Normalize temperature by dividing by 212
    // (highest possible temperature = 100°C = 212°F)
    return {
        input: [celsius / NORMALIZER],
        expected: (celsius * 1.8 + 32) / NORMALIZER
    };
});

const testData = data.splice(550, 50);

// 3. Train network
console.log("Training neural network...");

for (let i = 0; i < 800; i++) {
    shuffle(data).forEach(train =>
        network
            .populate(train.input)
            .run()
            .backpropagate([train.expected], 0.1, 1)
    );
}

let t1 = new Date().getTime();

console.log("Done Training");
console.log(`Took ${t1 - t0} milliseconds`);
console.log("-------------");

// 4. Test network
console.log("Testing neural network...");
console.log(
    "========================\n" +
        `Average error: ${testData
            .map(train => {
                let actual = network.predict(train.input)[0],
                    diff = Math.abs(train.expected - actual) * NORMALIZER;

                console.log(
                    `Converted ${train.input * NORMALIZER}°C to ${(
                        actual * NORMALIZER
                    ).toFixed(2)}°F instead of ${(
                        train.expected * NORMALIZER
                    ).toFixed(2)}°F`
                );

                return diff;
            })
            .reduce(plus, 0) / testData.length}`
);
