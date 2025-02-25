const tf = require('@tensorflow/tfjs-node');

// Define the game rules
const types = ['fire', 'grass', 'water', 'electric', 'ground'];
const typeAdvantages = {
  'fire': 'grass',
  'grass': 'water',
  'grass': 'ground',
  'water': 'fire',
  'water': 'ground',
  'ground': 'electric',
  'ground': 'fire',
  'electric': 'water',
};

// Function to generate training data
function generateTrainingData(numSamples) {
  const inputs = []; // Player's card (one-hot encoded)
  const labels = []; // AI's optimal card (one-hot encoded)

  for (let i = 0; i < numSamples; i++) {
    // Randomly select a player card
    const playerCard = types[Math.floor(Math.random() * types.length)];

    // One-hot encode the player's card
    const input = types.map(type => type === playerCard ? 1 : 0);
    inputs.push(input);

    // Determine the AI's optimal card based on the rules
    const aiCard = typeAdvantages[playerCard];
    const label = types.map(type => type === aiCard ? 1 : 0);
    labels.push(label);
  }

  return {
    inputs: tf.tensor2d(inputs),
    labels: tf.tensor2d(labels)
  };
}

// Generate 1000 samples of training data
const { inputs, labels } = generateTrainingData(1000);
console.log('Inputs:', inputs.arraySync());
console.log('Labels:', labels.arraySync());

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [types.length] })); // Input layer
model.add(tf.layers.dense({ units: types.length, activation: 'softmax' })); // Output layer

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Train the model
model.fit(inputs, labels, {
  epochs: 10,
  batchSize: 32,
  validationSplit: 0.2
}).then((history) => {
  console.log('Training complete!');
  console.log(history)
  console.log('Final accuracy:', history.history.acc.pop());

  // Save the model
  model.save('file://./ai_model').then(() => {
    console.log('Model saved to ai_model/');
  });
});