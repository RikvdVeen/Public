# ------------------------------------------------------------
# Binary flower classifier: Dandelion (0) vs Roses (4)
# Uses TensorFlow + EfficientNetB0 (transfer learning)
# ------------------------------------------------------------

# !pip install tensorflow tensorflow_datasets matplotlib
# !pip install tf-keras-vis

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

from google.colab import drive
drive.mount('/content/drive')


# reproducibility
SEED = 1
tf.random.set_seed(SEED)
np.random.seed(SEED)





#### First dataset
#### Positive: Sickle Cell
#### Negative: Normal Cell

# set train fraction for later use
train_frac = 0.9
# set epochs for later use
EPOCHS = 5

# ------------------------------------------------------------
# LOAD THE DATASET (full set)
# ------------------------------------------------------------
ds_loaded = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32, # Set batch_size here
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True
)

class_names = ds_loaded.class_names
print(class_names)

ds_filtered = ds_loaded

# ------------------------------------------------------------
# SPLIT (proportional)
# ------------------------------------------------------------
n = 0
for _ in ds_loaded:
    n += 1

train_size = int(n * train_frac)

# shuffle first so the split is random
ds_loaded = ds_loaded.shuffle(buffer_size=n, seed=SEED)

# Use tf.data operations for splitting
ds_train = ds_loaded.take(train_size)
ds_val = ds_loaded.skip(train_size)


print(f"Total: {n}, Train: {train_size}, Val: {n - train_size}")

# ------------------------------------------------------------
# PREPROCESS IMAGES
# ------------------------------------------------------------
IMG_SIZE = (256, 256)

def preprocess(images, labels):
    images = tf.image.resize(images, IMG_SIZE)
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels


ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.shuffle(1000).prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# Random checks (label and pixels)
# ------------------------------------------------------------
print("\nLabel check: sample labels (should be 0.0 or 1.0)")
for _, lbl in ds_train.unbatch().take(6):
    print(lbl.numpy())

print("\nPixels check: pixel min/max after preprocessing (one batch)")
for imgs, lbls in ds_train.take(1):
    print("images shape:", imgs.shape)
    print("min pixel:", tf.reduce_min(imgs).numpy())
    print("max pixel:", tf.reduce_max(imgs).numpy())
    break

# ------------------------------------------------------------
# MODEL DEFINITION (Transfer Learning)
# ------------------------------------------------------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(256, 256, 3), weights='imagenet'
)
base_model.trainable = False  

inputs = tf.keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)

# ------------------------------------------------------------
# Plotting of training curves
# ------------------------------------------------------------
hist = history.history

plt.figure(figsize=(12,4))

# Loss plot
plt.subplot(1,2,1)
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(alpha=0.3)

# Accuracy plot
plt.subplot(1,2,2)
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# EVALUATE MODEL
# ------------------------------------------------------------
eval_results = model.evaluate(ds_val)
print("\nValidation loss and accuracy:", eval_results)

# ------------------------------------------------------------
# CONFUSION MATRIX (cm)
# ------------------------------------------------------------
y_true, y_pred = [], []
for imgs, labels in ds_val.unbatch().batch(64):
    preds = model.predict(imgs)
    preds = (preds.flatten() >= 0.5).astype(int)
    y_pred.append(preds)
    y_true.append(labels.numpy().astype(int))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
TN, FP, FN, TP = cm.ravel()

# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------
print("Confusion matrix:")
print(cm)
print()

# Base counts
print(f"True Positive (TP): {TP}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Negative (TN): {TN}")
print()

# Derived measures
total = TP + TN + FP + FN
accuracy  = (TP + TN) / total if total else float('nan')
precision = TP / (TP + FP) if (TP + FP) else float('nan')
recall    = TP / (TP + FN) if (TP + FN) else float('nan')
specificity = TN / (TN + FP) if (TN + FP) else float('nan')
fpr = FP / (FP + TN) if (FP + TN) else float('nan')
fnr = FN / (FN + TP) if (FN + TP) else float('nan')

print("Derived measures:")
print(f"Accuracy              = (TP+TN)/(TP+FP+FN+TN) = {accuracy:.3f}")
print(f"Precision (PPV)       = TP/(TP+FP)            = {precision:.3f}")
print(f"Sensitivity           = TP/(TP+FN)            = {recall:.3f}")
print(f"Specificity           = TN/(TN+FP)            = {specificity:.3f}")
print(f"False Positive Rate   = FP/(FP+TN)            = {fpr:.3f}")
print(f"False Negative Rate   = FN/(FN+TP)            = {fnr:.3f}")


# ------------------------------------------------------------
# Grad-CAM heatmap generation
# ------------------------------------------------------------

def gradcam_heatmap(image_array_original, image_array_224_preprocessed, model, last_conv_layer_name="efficientnetb0", pred_index=None):
    # get the base EfficientNet model layer from the original model
    base_model_layer = model.get_layer(last_conv_layer_name)
    # get the last convolutional layer from the base model layer
    last_conv_layer = base_model_layer.get_layer("block7a_project_conv")
    print(f"Using layer: {last_conv_layer.name}")

    # Create a functional model on the fly that goes up to the last conv layer of the base model
    # This model takes a 224x224 input
    input_tensor = tf.keras.Input(shape=(256, 256, 3))
    x = input_tensor
    # Apply layers from base_model_layer up to the last_conv_layer, skipping the base model's original input
    apply_layers = False
    for layer in base_model_layer.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            apply_layers = True # Start applying layers after the InputLayer
            continue
        if apply_layers:
             x = layer(x)
             if layer.name == "block7a_project_conv":
                 break

    conv_model = tf.keras.models.Model(
        inputs=input_tensor,
        outputs=x # The output is the tensor after the last_conv_layer
    )


    # use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Watch the input image for gradients (the 224x224 version for conv_outputs)
        tape.watch(image_array_224_preprocessed)
        # Get the activations from the last convolutional layer using the temporary conv_model
        conv_outputs = conv_model(image_array_224_preprocessed, training=False)
        # Get the predictions from the original model using the original 256x256 image
        predictions = model(image_array_original, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # gradients of the predicted class with respect to the output feature map of conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ------------------------------------------------------------
# Visualization: overlay Grad-CAM heatmap on original image
# ------------------------------------------------------------
def display_gradcam(image_array, label, model):
    image_resized_224 = tf.image.resize(image_array, (224, 224))
    image_batch_224 = tf.expand_dims(image_resized_224, axis=0)
    image_batch_224_preprocessed = preprocess_input(image_batch_224) # Apply efficientnet preprocessing

    image_batch_original = tf.expand_dims(tf.image.resize(image_array, (256, 256)), axis=0)


    # get the heatmap
    heatmap = gradcam_heatmap(image_batch_original, image_batch_224_preprocessed, model)

    # create heatmap overlay
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    # Resize heatmap to the original image size for overlay
    image_resized_original = tf.image.resize(image_array, (256, 256)) # Resize to original model input size for overlay
    jet_heatmap = jet_heatmap.resize((image_resized_original.shape[1], image_resized_original.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)


    # superimpose heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + tf.cast(image_resized_original, tf.float32).numpy() # Overlay on 256x256 image
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # display images in a grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tf.cast(image_resized_original, tf.uint8)) # Display original 256x256 image
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')

    axes[2].imshow(superimposed_img)
    axes[2].set_title('Overlap')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Display Grad-CAM for a sample image from the validation set
# ------------------------------------------------------------
for image, label in ds_val.unbatch().take(1):
    display_gradcam(image, label.numpy(), model)
