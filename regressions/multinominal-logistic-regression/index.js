// require('@tensorflow/tfjs-node');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const loadData = () => {
  const mnistData = mnist.training(0, 10000);

  const features = mnistData.images.values.map((image) => _.flatMap(image));
  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels: encodedLabels };
};

const regression = (() => {
  const { features, labels } = loadData();

  return new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100,
  });
})();

regression.train();

const testMnistData = mnist.testing(0, 1000);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image),
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});
const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log(`Accuracy is ${accuracy}`);

plot({
  x: regression.costHistory.reverse(),
});
console.log(regression.costHistory);
