import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pygame

mnist= tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test)  =mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
 
 
 
 
 
 #Load the trained model
model = tf.keras.models.load_model('digits.keras')

loss, accuracy= model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 280  # 10x the MNIST size for easier drawing
DRAWING_SIZE = 28
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Handwritten Digit Recognition")
screen.fill(BLACK)

# Brush settings
brush_radius = 20

# Function to preprocess the image for prediction
def preprocess(surface):
    # Get pixel data from the Pygame surface
    data = pygame.surfarray.array3d(surface)
    # Convert to grayscale
    grayscale = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140]).T  # Transpose to align axes correctly
    # Invert colors to match MNIST (white digit on black background)
    inverted = 255 - grayscale
    # Resize to 28x28 (MNIST size)
    resized_surface = pygame.transform.smoothscale(
        pygame.surfarray.make_surface(np.stack([inverted] * 3, axis=-1)),
        (DRAWING_SIZE, DRAWING_SIZE)
    )
    # Convert resized surface to array and take a single channel
    resized_array = pygame.surfarray.array3d(resized_surface).mean(axis=-1)  # Convert to single channel
    # Normalize to [0, 1]
    normalized = resized_array.astype(np.float32) / 255.0
    # Reshape to match model input (batch_size, height, width, channels)
    normalized = normalized.reshape(1, DRAWING_SIZE, DRAWING_SIZE, 1)

    # Save preprocessed image for debugging (optional)
    plt.imsave("debug_preprocessed_digit.png", resized_array, cmap='gray')

    return normalized

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Handle mouse input for drawing
        elif event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:  # Left mouse button is held
                pygame.draw.circle(screen, WHITE, event.pos, brush_radius)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Predict when SPACE is pressed
                input_image = preprocess(screen)
                
                # Visualize the preprocessed input
                plt.imshow(input_image[0, :, :, 0], cmap='gray')
                plt.title("Preprocessed Input")
                plt.show()

                # Predict digit
                prediction = model.predict(input_image)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0, predicted_digit]
                print(f"Predicted Digit: {predicted_digit} with Confidence: {confidence:.2f}")
            elif event.key == pygame.K_c:  # Clear the screen when 'C' is pressed
                screen.fill(BLACK)

    pygame.display.flip()

pygame.quit()