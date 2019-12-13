const outputs = [];
const predictionPoint = 300;
const distance = (pointA, pointB) => Math.abs(pointA - pointB);
const k = 3;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  // Write code here to analyze stuff
  const testSetSize = 10;
  const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

  const accuracy = _.chain(testSet)
    .filter(testPoint => knn(trainingSet, testPoint[0]) === testPoint[3])
    .size()
    .divide(testSetSize)
    .value();
  console.log('Accuracy is ' + accuracy);
}

function knn(data, point) {
  return _.chain(data)
    .map(row => [distance(row[0], point), row[3]])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}
