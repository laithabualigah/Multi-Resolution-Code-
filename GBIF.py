import numpy as np
import networkx as nx
from skimage import io, img_as_float
from skimage.util import random_noise
from skimage.transform import resize

# Data Pre-processing
def preprocess_image(image_path, target_size=(128, 128)):
    """Load and resize an image to the target size."""
    image = img_as_float(io.imread(image_path))
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image_resized = resize(image, target_size, anti_aliasing=True)
    return image_resized

def add_noise(image, noise_type='gaussian', var=0.01):
    """Add noise to the image."""
    noisy_image = random_noise(image, mode=noise_type, var=var)
    return noisy_image

# Graph-Based Image Filtering (GBIF)
def build_image_graph(image):
    """Construct a graph from the image pixels."""
    height, width, _ = image.shape
    graph = nx.grid_2d_graph(height, width)

    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            node = (i, j)
            graph.nodes[node]['value'] = pixel_value

            # Connect edges with weights based on pixel similarity
            if i > 0:
                graph.add_edge((i, j), (i - 1, j), weight=np.linalg.norm(pixel_value - image[i - 1, j]))
            if j > 0:
                graph.add_edge((i, j), (i, j - 1), weight=np.linalg.norm(pixel_value - image[i, j - 1]))
    return graph

def graph_filter(graph, alpha=0.85):
    """Apply graph-based smoothing to the image graph."""
    new_values = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_weights = [graph[node][neighbor]['weight'] for neighbor in neighbors]
        neighbor_values = [graph.nodes[neighbor]['value'] for neighbor in neighbors]

        smoothed_value = alpha * graph.nodes[node]['value'] + (1 - alpha) * np.mean(neighbor_values, axis=0)
        new_values[node] = smoothed_value

    for node, value in new_values.items():
        graph.nodes[node]['value'] = value

    return graph

def reconstruct_image(graph, shape):
    """Reconstruct the image from the graph nodes."""
    height, width, _ = shape
    reconstructed_image = np.zeros(shape)

    for i in range(height):
        for j in range(width):
            reconstructed_image[i, j] = graph.nodes[(i, j)]['value']

    return np.clip(reconstructed_image, 0, 1)

if __name__ == "__main__":
    # Example usage
    image_path = "example_image.jpg"  # Replace with your image path
    image = preprocess_image(image_path)
    noisy_image = add_noise(image)

    # Build graph from the noisy image
    print("Building image graph...")
    image_graph = build_image_graph(noisy_image)

    # Apply graph-based image filtering
    print("Applying GBIF...")
    filtered_graph = graph_filter(image_graph)

    # Reconstruct the image
    print("Reconstructing filtered image...")
    filtered_image = reconstruct_image(filtered_graph, noisy_image.shape)

    # Save or display the output
    io.imsave("filtered_image.jpg", (filtered_image * 255).astype(np.uint8))
    print("Filtered image saved as 'filtered_image.jpg'")
