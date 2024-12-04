import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pygame

mnist= tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test)  =mnist.load_data()


 
 
 
 
 
 
model = tf.keras.models.load_model('digits2.keras')

loss, accuracy= model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

pygame.init()


DRAWING_SIZE = 28  
SCALE_FACTOR = 10  
WINDOW_SIZE = DRAWING_SIZE * SCALE_FACTOR
BRUSH_RADIUS = SCALE_FACTOR // 2 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Digit Drawing (28x28 scaled)")


screen.fill(BLACK)


def preprocess(surface):
    
    data = pygame.surfarray.array3d(surface)
   
    grayscale = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140]).T
  
    inverted = 255 - grayscale
   
    resized = pygame.transform.smoothscale(
        pygame.surfarray.make_surface(np.stack([inverted] * 3, axis=-1)),
        (DRAWING_SIZE, DRAWING_SIZE)
    )
   
    resized_array = pygame.surfarray.array3d(resized).mean(axis=-1)
   
    normalized = resized_array.astype(np.float32) / 255.0
  
    normalized = normalized.reshape(1, DRAWING_SIZE, DRAWING_SIZE, 1)

   
    return normalized, resized_array




running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]: 
                pygame.draw.circle(screen, WHITE, event.pos, BRUSH_RADIUS)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  
                input_image, resized_array = preprocess(screen)
                
             
                plt.imshow(resized_array, cmap='gray')
                plt.title("Preprocessed Input (28x28)")
                plt.colorbar()
                plt.show()

              
                prediction = model.predict(input_image)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0, predicted_digit]
                print(f"Predicted Digit: {predicted_digit} with Confidence: {confidence:.2f}")
            elif event.key == pygame.K_c:  
                screen.fill(BLACK)

   
    pygame.display.flip()

pygame.quit()
