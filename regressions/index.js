require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100,
});

regression.train();

console.log(
  'Update M is: ',
  regression.weights.arraySync()[1][0],
  'Updated B is: ',
  regression.weights.arraySync()[0][0],
);
