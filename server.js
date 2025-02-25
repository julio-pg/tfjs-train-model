const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
const port = 8000;

let model;

// Load the model
tf.loadLayersModel('file://./saved_model/model.json').then(loadedModel => {
  model = loadedModel;
  console.log('Model loaded');
});

app.get('/predict', async (req, res) => {
  const input = parseFloat(req.query.input);
  if (isNaN(input)) {
    return res.status(400).send('Invalid input');
  }

  const tensor = tf.tensor2d([input], [1, 1]);
  const prediction = model.predict(tensor);
  const output = prediction.dataSync()[0];

  res.send({ input, output });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});