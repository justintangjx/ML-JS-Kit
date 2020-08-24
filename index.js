require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

let knn = (features, labels, predictionPoint, k) => {
  return features
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
    .slice(0, k)
    .reduce((acc, pair) => acc + pair.get(1), 0) / k;
};

// load data on separate csv
let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    // extract only these 2 parameters; location
    dataColumns: ["lat", "long"],
    // only to predict price label
    labelColumns: ["price"],
  }
);

// run analysis
// convert arrays to tensors first...
features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('error', err * 100);
})
// const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

console.log();
//