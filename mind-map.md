Core Machine Learning Concepts
│
├── Machine Learning Model
│   ├── Definition: Model = Training (Algorithm + Data)
│   ├── Purpose: Learn patterns from data
│   └── Function: Map inputs → outputs/predictions
│
├── Model Types
│   ├── Linear Models
│   │   ├── Linear Regression (continuous outcomes)
│   │   └── Logistic Regression (probability/classification)
│   ├── Tree-Based Models
│   │   ├── Decision Trees (hierarchical rules)
│   │   └── Random Forest (ensemble of trees)
│   ├── Distance-Based
│   │   └── Support Vector Machines (hyperplane separation)
│   ├── Probabilistic
│   │   └── Naive Bayes (Bayes' theorem)
│   └── Deep Learning
│       └── Neural Networks (interconnected neurons)
│
├── Statistical Methods
│   ├── Ordinary Least Squares (OLS)
│   │   ├── Minimizes squared differences
│   │   ├── Assumes normal distribution
│   │   └── Constant variance (homoscedasticity)
│   └── Objective Functions
│       ├── Also called: Cost/Loss/Optimization function
│       └── Goal: Minimize or maximize performance
│
├── Optimization
│   ├── Optimizers
│   │   ├── Purpose: Adjust model parameters
│   │   └── Goal: Minimize loss function
│   ├── Gradient Descent
│   │   └── Iterative parameter adjustment
│   └── Backpropagation
│       └── Compute gradients in neural networks
│
├── Performance Concepts
│   ├── Overfitting
│   │   ├── Good on training data
│   │   └── Poor on new data
│   ├── Underfitting
│   │   └── Too simple to capture patterns
│   ├── Bias-Variance Tradeoff
│   │   ├── Low bias: Fits training data well
│   │   └── Low variance: Generalizes well
│   └── Regularization
│       └── Prevents overfitting with constraints
│
├── Training Process
│   ├── Loss Function
│   │   └── Measures prediction accuracy
│   ├── Hyperparameters
│   │   ├── Set before training
│   │   ├── Learning Rate
│   │   └── Number of layers
│   ├── Cross-Validation
│   │   └── Assess model performance
│   └── Techniques
│       ├── Dropout (deactivate neurons)
│       ├── Batch Size (examples per iteration)
│       └── Bias (shift output non-linearly)
│
├── Data Processing
│   ├── Feature Extraction
│   │   └── Select/transform relevant features
│   ├── Inference
│   │   └── Use trained model on new data
│   └── Quantization
│       └── Reduce precision for efficiency
│
└── Technical Terminology
    ├── Slug: Human-readable identifier
    ├── Token: Unit of text
    ├── Chain-of-thought: Sequential reasoning flow
    └── Activation Function: Introduce non-linearity