const tf = require('@tensorflow/tfjs');

class LogisticRegression {
  constructor(features, labels, options) {
    // For standardization
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    const filler = variance.cast('bool').logicalNot().cast('float32');
    this.variance = variance.add(filler);

    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = []; // Cross entropy
    this.bHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 },
      options,
    );

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize,
    );

    for (let i = 0; i < this.options.iterations; ++i) {
      this.bHistory.push(this.weights.arraySync()[0][0]);
      for (let j = 0; j < batchQuantity; ++j) {
        const { batchSize } = this.options;
        const startIndex = j * batchSize;

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1],
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1],
          );

          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(testLabels).sum().arraySync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standardize(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();

      const termOne = this.labels.transpose().matMul(guesses.log());

      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).log());

      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync()[0];
    });

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
