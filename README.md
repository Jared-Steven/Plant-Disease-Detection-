# Plant Disease Prediction Using Convolutional Neural Networks

This repository contains code for a Convolutional Neural Network (CNN) model to predict plant diseases using images. The model is trained on a dataset containing images of diseased plants and normal plants. Below is a brief explanation of the code structure and its functionalities:
Dependencies

    numpy: For numerical operations.
    pandas: For data manipulation and analysis.
    matplotlib: For data visualization.
    opencv-python: For image processing.
    scikit-learn: For machine learning utilities like data splitting and label binarization.
    PIL: Python Imaging Library for opening, manipulating, and saving many different image file formats.
    keras: Deep learning library for creating and training neural networks.
    tensorflow: Backend for Keras, providing support for building deep learning models.

Overview of the Code

    Data Visualization
        12 random images from the dataset are plotted using Matplotlib to observe the images.

    Image Preprocessing
        Images are converted into numpy arrays and resized to a common size.
        The dataset is normalized by dividing pixel values by 255.

    Data Loading and Labeling
        Images are loaded into arrays along with their corresponding labels.
        Labels are encoded using one-hot encoding.

    Model Architecture
        A CNN model is defined using Keras Sequential API.
        The model consists of Convolutional layers, MaxPooling layers, Flatten layer, and Dense layers.

    Model Training
        The dataset is split into training, validation, and testing sets.
        The model is trained on the training set and validated on the validation set.
        Training history is recorded to monitor the model's performance over epochs.

    Model Evaluation and Saving
        The trained model is evaluated on the testing set to calculate accuracy.
        The model, its architecture, and weights are saved for future use.

    Result Visualization
        The accuracy of the model over epochs is plotted.
        An example image from the testing set is shown along with its original and predicted labels.

Running the Code

To run the code:

    Ensure all dependencies are installed.
    Modify the paths to the dataset according to your directory structure.
    Execute the code sequentially.

This README.md provides an overview of the project and its functionalities. For more detailed information, refer to the comments within the code.
