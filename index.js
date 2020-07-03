const fs = require('fs');
const CSV = require('comma-separated-values');
const trainTestSplit = require('train-test-split');
const ConfusionMatrix = require('ml-confusion-matrix');
const _ = require('lodash');
var Model = require('machine_learning').MLP;

SLICE_INDEX = 500

function roll(v) {
  return _.indexOf(v, _.max(v));
}

function unroll(v) {
  let t = _.fill(Array(10), 0);
  
  t[v] = 1;

  return t;
}

let allData = CSV.parse(fs.readFileSync(('./data/train.csv'), 'utf-8')).slice(1, SLICE_INDEX)
let [trainData, validationData] = trainTestSplit(allData, 0.8, 1234)

let yTrainData = _.map(trainData, (d) => Number(d[0]))
let xTrainData = _.map(trainData, (d) => _.slice(d, 1))

let yValidationData = _.map(validationData, (d) => Number(d[0]))
let xValidationData = _.map(validationData, (d) => _.slice(d, 1))

xTrainData = _.map(xTrainData, (row) => _.map(row, (value) => value / 255))
xValidationData = _.map(xValidationData, (row) => _.map(row, (value) => value / 255))

let model = new Model({
  'input': xTrainData,
  'label': _.map(yTrainData, (value) => unroll(value)),
  'n_ins': 784,
  'n_outs': 10,
  'hidden_layer_sizes': [100] 
});

model.set('log level', 1);
model.train({ 'lr': 0.05, 'epochs' : 1000 });

let yResult = model.predict(xValidationData)
let CM2 = ConfusionMatrix.fromLabels(yValidationData, _.map(yResult, (row) => roll(row)));

console.log('Accuracy: ' + CM2.getAccuracy());
