# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 30
IMAGE_SIZE = (100, 100) # Define image size as a variable

# --- THIS IS THE BIGGEST CHANGE ---
# 1. Add Data Augmentation to the training generator to prevent overfitting
# 2. Add validation_split to automatically create a test set (20% of data)

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of data for validation
)

# The validation generator should NOT have augmentation, just rescaling
validation_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# --- CORRECTED FILE PATH FOR COLAB ---
# Change this path to where you unzipped your folder in Colab
data_dir = '/content/Diabetic' 

# Flow training images in batches using train_datagen
train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        classes=['Diabetics Not Affected', 'Diabetics Affected'],
        class_mode='categorical',
        subset='training'  # Specify this is the training subset
)

# Flow validation images
validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        classes=['Diabetics Not Affected', 'Diabetics Affected'],
        class_mode='categorical',
        subset='validation'  # Specify this is the validation subset
)

# Build the CNN model
model = Sequential([
    Convolution2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    
    Convolution2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Adding one more conv layer
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    # Adding Dropout layer to prevent overfitting
    Dropout(0.5),
    Dense(128, activation='relu'),
    # 2 output neurons for 2 classes
    Dense(2, activation='softmax')
])

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])  # 'acc' is short for accuracy

# Number of epochs
n_epochs = 30

# Train the model using both train and validation generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=n_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=1
)   

# --- PLOTS ARE UPDATED TO SHOW VALIDATION METRICS ---

# Plot training & validation accuracy
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Save the model
model.save('diabetic_retinopathy_model.h5')