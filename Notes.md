# Misc Notes

### Slug
A slug typically refers to a short, human-readable text string that is used to identify a specific resource, such as 
a dataset, model, or project. It is often used in URLs or file names to provide a concise and descriptive identifier. 
For example, a slug for a machine learning project on image recognition could be "image-recognition-project". Slugs 
are useful for organizing and referencing resources in a more user-friendly manner.

### Quantization 
Quantization refers to the process of reducing the precision or number of bits used to represent numerical values. 
It involves converting continuous values into a discrete set of values. Quantization is often used to compress models 
or reduce memory usage, making them more efficient for deployment on resource-constrained devices. For example, instead 
of using 32-bit floating-point numbers, quantization may reduce them to 8-bit integers. This can result in faster 
computations and reduced storage requirements, albeit with a slight loss in precision.

### Inference
Inference refers to the process of using a trained model to make predictions or draw conclusions based on new, unseen 
data. It involves applying the learned knowledge from the model to make decisions or generate outputs. For example, if 
a model is trained to classify images of cats and dogs, inference would involve using the trained model to predict whether 
a new image contains a cat or a dog. Inference is a crucial step in utilizing machine learning models for real-world applications.

### Supervised Learning
A type of ML where the model learns from labeled data to make predictions or classifications.

### Unsupervised Learning
ML technique where the model learns patterns and structures from unlabeled data without specific guidance.

### Overfitting
When a model performs well on training data but fails to generalize to new, unseen data due to excessive complexity.

### Underfitting
When a model is too simple to capture the underlying patterns in the data, resulting in poor performance.

### Regularization
Techniques used to prevent overfitting by adding additional constraints or penalties to the model during training.

### Feature Extraction
The process of selecting or transforming relevant features from raw data to improve model performance.

### Loss Function
A measure of how well a model's predictions align with the actual values, used to optimize the model during training.

### Gradient Descent
An optimization algorithm that adjusts the model's parameters iteratively to minimize the loss function.

### Bias-Variance Tradeoff
The balance between a model's ability to fit the training data (low bias) and its ability to generalize to new data (low variance). Finding the right balance is crucial to avoid underfitting or overfitting.

### Hyperparameters
Parameters set before training that influence the model's learning process, such as learning rate or number of layers.

### Cross-Validation
A technique to assess model performance by splitting data into multiple subsets for training and evaluation.

### Ensemble Learning
A technique that combines multiple models, known as an ensemble, to make predictions or classifications. It often leads to improved performance and robustness.

### Transfer Learning
The practice of leveraging knowledge gained from training one model on a specific task to improve the performance of a different but related task.

### Reinforcement Learning
A type of Machine Learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.

### Convolutional Neural Networks (CNN)
Neural networks specifically designed for processing grid-like data, such as images or audio, by using convolutional layers to extract relevant features.

### Recurrent Neural Networks (RNN)
Neural networks that can process sequential data by utilizing feedback connections, making them suitable for tasks like natural language processing and speech recognition.

### Activation Function
A mathematical function applied to the output of a neuron in a neural network, introducing non-linearity and enabling complex mappings between inputs and outputs.

### Batch Size
The number of training examples used in each iteration of the training process. Larger batch sizes can lead to faster training, but may require more memory.

### Learning Rate
A hyperparameter that determines the step size at which the model's parameters are updated during training. It influences the speed and stability of the learning process.

### Dropout
A regularization technique used during training to randomly deactivate a certain percentage of neurons in a neural network, reducing overfitting and improving generalization.

### Backpropagation
An algorithm used to compute the gradients of the loss function with respect to the parameters of a neural network, enabling efficient parameter updates during training.

### Bias
An additional parameter in a machine learning model that allows for shifting the output in a non-linear manner, providing flexibility and improving model performance.

### Chain-of-thought
It refers to the sequential flow of reasoning or decision-making processes within a model. It represents how information is processed and transformed as it moves through different layers or components of the model.

### Token
A token refers to a unit of text that is treated as a single entity. It can be a word, a character, or even a subword, depending on how the text is processed. For example, in natural language processing tasks, a sentence can be tokenized into individual words or subwords, which are then used as inputs for various algorithms and models. Tokenization helps in organizing and analyzing text data in a structured manner.