# Digit-Recognition-using-CNN-Keras-
Digit Recognition using CNN (Keras)
1. Library Import and Environment Setup
Code Function:
This section imports the essential Python libraries required for data manipulation and model building:
•	NumPy & Pandas: Used for linear algebra operations and reading the dataset (CSV files).
•	Seaborn & Matplotlib: Used for visualizing data distributions and plotting graphs.
•	Warnings: configured to ignore warning messages to keep the output clean.
•	OS: Used to list the files in the input directory to verify the data path.
2. Data Loading and Separation	
Code Function:
This section handles the ingestion of the MNIST digit dataset:
•	The code reads train.csv and test.csv into Pandas DataFrames.
•	Feature Separation: It separates the target labels (numbers 0-9) into a variable Y_train and the pixel data (features) into X_train.
•	It drops the "label" column from the training set so that X_train contains only image pixel data.
3. Exploratory Data Analysis (Visualization)
Code Function:
Before processing, the code visualizes the dataset to understand its structure:
•	Count Plot: Uses sns.countplot to display the frequency of each digit (0-9) in the training set. This checks if the dataset is balanced (i.e., if there are equal numbers of 1s, 2s, etc.).
•	Image Sampling: Converts the flat pixel data of specific rows (e.g., row 0 and row 3) into 28x28 matrices and displays them using plt.imshow. This confirms that the data corresponds to recognizable handwritten digits.
4. Data Preprocessing
Code Function:
This step prepares the raw data for the Neural Network :
•	Normalization: The pixel values (0–255) are divided by 255.0 to scale them to a range of 0–1. This helps the CNN converge faster.
•	Reshaping: The flat rows of pixels are reshaped into 3D matrices with dimensions (28, 28, 1). This is necessary because Keras requires the image height, width, and channel (1 for grayscale).
•	Label Encoding: The integer labels (e.g., "5") are converted into One-Hot Encoded vectors (e.g., [0,0,0,0,0,1,0,0,0,0]) using to_categorical.
5. Train-Validation Split
Code Function:
To evaluate the model effectively, the training data is split into two parts using train_test_split:
•	Training Set (90%): Used to teach the model.
•	Validation Set (10%): Used to test the model's accuracy during training on data it hasn't seen before.
6. Convolutional Neural Network (CNN) Architecture
Code Function:
The model is built using the Keras Sequential API with the following layers :
1.	Conv2D Layer 1: 8 filters of size (5x5) with ReLU activation to detect basic features (edges).
2.	MaxPool2D: Reduces the image size to lower computational cost.
3.	Dropout (0.25): Randomly disables 25% of neurons to prevent overfitting.
4.	Conv2D Layer 2: 16 filters of size (3x3) with ReLU activation to detect complex features.
5.	MaxPool2D & Dropout: Further down-sampling and regularization.
6.	Flatten: Converts the 2D feature maps into a 1D vector.
7.	Dense (Fully Connected): A layer of 256 neurons.
8.	Output Layer: A Dense layer with 10 neurons (one for each digit) using the Softmax activation function to output probabilities.
7. Optimizer and Compilation
Code Function:
•	Optimizer: Defines the Adam optimizer with a specific learning rate (0.001) to update the weights during training.
•	Compile: Configures the model to use categorical_crossentropy (the standard loss function for multi-class classification) and tracks the accuracy metric.
8. Data Augmentation
Code Function:
To improve the model's generalization, ImageDataGenerator is used to artificially expand the dataset. It applies random transformations to the training images:
•	Rotation (5 degrees)
•	Zooming (10%)
•	Width and Height shifting (10%)
•	Note: It does not flip images vertically or horizontally, as upside-down numbers (like 6 and 9) could be confusing.
9. Model Training
Code Function:
The model is trained using fit_generator:
•	It feeds the augmented data batches into the model.
•	Epochs: The model iterates through the dataset 10 times.
•	Batch Size: It processes 250 images at a time.
•	It validates performance against (X_val, Y_val) after every epoch.
10. Performance Evaluation
Code Function:
After training, the code assesses the results:
•	Loss Curve: Plots the Training Loss vs. Validation Loss to check for overfitting .
•	Confusion Matrix: Calculates predictions on the validation set and plots a Heatmap. This shows exactly which numbers are being confused with others (e.g., how often a "4" is predicted as a "9").

