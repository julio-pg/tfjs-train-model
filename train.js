const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Define a simple model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compile the model
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Generate some synthetic data for training
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model
model.fit(xs, ys, { epochs: 10 }).then(() => {
  console.log('Model training complete!');

  // Save the model with an incremental name

  const dir = './saved_model';
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }

  const files = fs.readdirSync(dir);
  const modelNumber = files.length + 1;
  const modelPath = path.join(dir, `model-${modelNumber}`);

  model.save(`file://${modelPath}`).then(() => {
    console.log(`Model saved to ${modelPath}/`);
  });

});