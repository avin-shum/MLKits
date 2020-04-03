// require('@tensorflow/tfjs-node');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');
const mnist = require('mnist-data');

const mnistData = mnist.training(0, 10);

const features = mnistData.images.values.map((image) => _.flatMap(image));

console.log(features);
