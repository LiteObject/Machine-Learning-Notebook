# Machine Learning Terminology

## Neural Networks
- **Artificial Neural Network**: Deep learning models composed of interconnected layers of artificial neurons
- **Convolutional Neural Network (CNN)**: Neural networks specifically designed for processing grid-like data, such as images or audio, by using convolutional layers to extract relevant features
- **Recurrent Neural Network (RNN)**: Neural networks that can process sequential data by utilizing feedback connections, making them suitable for tasks like natural language processing and speech recognition
- **Long Short-Term Memory (LSTM)**: A type of RNN that can learn long-term dependencies
- **Generative Adversarial Network (GAN)**: A framework where two neural networks compete against each other

## Learning Approaches
- **Supervised Learning**: A type of ML where the model learns from labeled data to make predictions or classifications
- **Unsupervised Learning**: ML technique where the model learns patterns and structures from unlabeled data without specific guidance
- **Semi-Supervised Learning**: Learning approach that uses both labeled and unlabeled data
- **Reinforcement Learning**: A type of Machine Learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties
- **Transfer Learning**: The practice of leveraging knowledge gained from training one model on a specific task to improve the performance of a different but related task
- **Ensemble Learning**: A technique that combines multiple models, known as an ensemble, to make predictions or classifications. It often leads to improved performance and robustness

## Model Evaluation
- **Overfitting**: When a model performs well on training data but fails to generalize to new, unseen data due to excessive complexity
- **Underfitting**: When a model is too simple to capture the underlying patterns in the data, resulting in poor performance
- **Bias**: An additional parameter in a machine learning model that allows for shifting the output in a non-linear manner, providing flexibility and improving model performance
- **Variance**: The model's sensitivity to fluctuations in the training data
- **Loss Function**: A measure of how well a model's predictions align with the actual values, used to optimize the model during training
- **Regularization**: Techniques used to prevent overfitting by adding additional constraints or penalties to the model during training

## Optimization Techniques
- **Gradient Descent**: An optimization algorithm that adjusts the model's parameters iteratively to minimize the loss function
- **Backpropagation**: An algorithm used to compute the gradients of the loss function with respect to the parameters of a neural network, enabling efficient parameter updates during training
- **Dropout**: A regularization technique used during training to randomly deactivate a certain percentage of neurons in a neural network, reducing overfitting and improving generalization
- **Batch Normalization**: A technique to normalize inputs to each layer
- **Learning Rate**: A hyperparameter that determines the step size at which the model's parameters are updated during training. It influences the speed and stability of the learning process
- **Hyperparameter**: Parameters set before training that influence the model's learning process, such as learning rate or number of layers

## Data Processing
- **Feature Extraction**: The process of selecting or transforming relevant features from raw data to improve model performance
- **Feature Selection**: The process of selecting the most relevant features for model training
- **Dimensionality Reduction**: Techniques to reduce the number of features while preserving important information
- **Word Embedding**: Dense vector representations of words that capture semantic meaning

## Algorithms
- **Support Vector Machine (SVM)**: Models data by finding a hyperplane that best separates different classes or groups
- **Decision Tree**: Models decisions or classifications by splitting the data based on a series of hierarchical rules or conditions
- **Random Forest**: An ensemble model that combines multiple decision trees to make predictions by averaging or voting
- **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies data points based on the majority class of their k nearest neighbors
- **Clustering**: Unsupervised learning technique that groups similar data points together

## Natural Language Processing (NLP)
- **Natural Language Processing (NLP)**: Field of AI focused on enabling computers to understand, interpret, and generate human language
- **Sentiment Analysis**: The process of determining the emotional tone or opinion expressed in text
- **Token**: A unit of text that is treated as a single entity. It can be a word, a character, or even a subword, depending on how the text is processed
- **Chain-of-thought**: The sequential flow of reasoning or decision-making processes within a model

## General Terms
- **Slug**: A short, human-readable text string that is used to identify a specific resource, such as a dataset, model, or project
- **Inference**: The process of using a trained model to make predictions or draw conclusions based on new, unseen data
- **Quantization**: The process of reducing the precision or number of bits used to represent numerical values
- **Activation Function**: A mathematical function applied to the output of a neuron in a neural network, introducing non-linearity
- **Batch Size**: The number of training examples used in each iteration of the training process
- **Cross-Validation**: A technique to assess model performance by splitting data into multiple subsets for training and evaluation
