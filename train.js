const tf = require('@tensorflow/tfjs-node');

// Define the game rules
const types = ['fire', 'grass', 'water', 'electric', 'ground'];
const typeAdvantages = {
  'fire': 'water',
  // 'fire': 'ground',
  'grass': 'fire',
  'water': 'electric',
  // 'water': 'grass',
  'electric': 'ground',
  'ground': 'grass',
  // 'ground': 'water',
};

// Function to generate training data
function generateTrainingData(numSamples) {
  const inputs = [];
  const labels = [];

  for (let i = 0; i < numSamples; i++) {
    // Randomly select player's active monster
    const playerActiveMonster = types[Math.floor(Math.random() * types.length)];
    const playerIndex = types.indexOf(playerActiveMonster);
    const playerIndexNormalized = playerIndex / (types.length - 1);

    // Generate AI's available monsters
    const aiAvailableMonsters = new Array(types.length).fill(0);
    const numAvailable = Math.floor(Math.random() * (types.length + 1));
    const availableIndices = new Set();
    while (availableIndices.size < numAvailable) {
      availableIndices.add(Math.floor(Math.random() * types.length));
    }
    availableIndices.forEach(index => aiAvailableMonsters[index] = 1);

    // Combine into input
    const input = [playerIndexNormalized, ...aiAvailableMonsters];
    inputs.push(input);

    // Determine AI's card
    const optimalType = typeAdvantages[playerActiveMonster];
    const optimalIndex = types.indexOf(optimalType);
    const isOptimalAvailable = aiAvailableMonsters[optimalIndex] === 1;

    let aiCard;

    if (isOptimalAvailable) {
      aiCard = optimalType;
    } else {
      // Collect non-weak candidates (excluding optimalType)
      const fallbackCandidates = [];
      for (let i = 0; i < types.length; i++) {
        if (aiAvailableMonsters[i] === 1 && types[i] !== optimalType) {
          fallbackCandidates.push(types[i]);
        }
      }
      if (fallbackCandidates.length > 0) {
        // Select the first candidate (alternatively, randomize)
        aiCard = fallbackCandidates[0];
      } else {
        // Select any available as fallback (even if weak)
        for (let i = 0; i < aiAvailableMonsters.length; i++) {
          if (aiAvailableMonsters[i] === 1) {
            aiCard = types[i];
            break;
          }
        }
        // Handle edge case where no monsters are available (unlikely due to numAvailable)
        if (!aiCard) aiCard = types[0]; // default fallback
      }
    }

    // Encode label
    const label = types.map(type => type === aiCard ? 1 : 0);
    labels.push(label);
    console.log('playerMonster:', playerActiveMonster)
    console.log('aiMonster:', aiCard)
  }

  return {
    inputs: tf.tensor2d(inputs, [numSamples, types.length + 1]),
    labels: tf.tensor2d(labels, [numSamples, types.length])
  };
}


// Generate 1000 samples of training data
const { inputs, labels } = generateTrainingData(3000);
// console.log('Inputs:', inputs.arraySync());
// console.log('Labels:', labels.arraySync());

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [6] })); // Input layer
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
  console.log('Final accuracy:', history.history.acc.pop());

  // Save the model
  model.save('file://./ai_model').then(() => {
    console.log('Model saved to ai_model/');
  });
});