
# TensorFlow.js Node.js Project

This repository contains a simple Node.js project that demonstrates how to train an AI model using TensorFlow.js, save the trained model, and serve predictions via an Express server.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Serving Predictions](#serving-predictions)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates how to:
1. Train a simple linear regression model using TensorFlow.js in a Node.js environment.
2. Save the trained model to the filesystem.
3. Load the saved model and serve predictions via an Express server.

## Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v14 or higher)
- [npm](https://www.npmjs.com/) (usually comes with Node.js)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

## Usage

### Training the Model

To train the model, run the following command:
```bash
node train.js
```

This script will:
- Define a simple linear regression model.
- Train the model using synthetic data.
- Save the trained model to the `ai_model/` directory.

### Serving Predictions

To serve predictions using the trained model, start the Express server:
```bash
node server.js
```

The server will start on `http://localhost:3000`. You can make a prediction by sending a GET request to the `/predict` endpoint with an `input` query parameter:
```
http://localhost:3000/predict?input=5
```

Example response:
```json
{
  "input": 5,
  "output": 9
}
```

## Saving and Loading the Model

The trained model is saved in the `saved_model/` directory. The model consists of:
- `model.json`: The model architecture and weights manifest.
- `weight files`: Binary files containing the model weights.

To load the model in another script or application, use the following code:
```javascript
const tf = require('@tensorflow/tfjs-node');

tf.loadLayersModel('file://./saved_model/model.json').then(model => {
    console.log('Model loaded successfully!');
    // Use the model for predictions or further training
});
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.