from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def quantize_image(image, decision_levels, representation_levels):
    # Initialize quantized image
    quantized_image = np.zeros_like(image)

    # Iterate through each pixel and quantize
    for i in range(len(decision_levels) - 1):
        mask = (image >= decision_levels[i]) & (image < decision_levels[i + 1])
        quantized_image[mask] = representation_levels[i]

    # Handle the last interval separately
    mask = image >= decision_levels[-1]
    quantized_image[mask] = representation_levels[-1]
    return quantized_image

def load_image_as_array(image_path):
    # load the image into numpy array
    with Image.open(image_path) as img:
        img_array = np.array(img)
    return img_array.flatten()

def create_histogram(img_arr):
    hist = np.zeros(256)
    for pixel in img_arr:
        hist[pixel] += 1
    return hist

def show_histogram(pdf):
    indexes = np.arange(len(img_arr))
    plt.bar(indexes, pdf , color='skyblue', edgecolor='black', width=0.5)
    plt.title('Histogram with Array Indexes as X-Axis')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def uniform_quantization(image, b):
    levels = 2**b
    quantization_step = (image.max() - image.min()) / levels
    quantized_image = np.zeros_like(image, dtype=np.uint8)
    interval_entries_array = np.floor((image - image.min())/quantization_step)
    # in each interval, find representative quantizer:
    quantized_image = image.min() + (interval_entries_array+0.5)*quantization_step
    mse = np.mean((image - quantized_image)**2)
    return quantized_image, mse

def perform_quantization_for_bits(image, bit_budgets):
    mse_values = np.zeros_like(bit_budgets, dtype=float)
    q_images = []
    # Perform uniform quantization for each bit budget
    for i, b in enumerate(bit_budgets):
        quantized_image, mse = uniform_quantization(image, b)
        q_images.append(quantized_image)
        mse_values[i] = mse
    print(mse_values)
    return mse_values, q_images

def show_bit_quantization_plot(bit_budgets, mse_values):
    plt.figure()
    plt.plot(bit_budgets, mse_values, marker='o', linestyle='-', color='b')
    plt.title('MSE vs Bit-Budget (b)')
    plt.xlabel('Bit-Budget (b)')
    plt.ylabel('MSE')
    plt.xticks(bit_budgets)
    plt.grid(True)
    plt.show()

def calculate_representation_levels(hist, decision_levels):
  prob = hist/hist.sum()
  x_prob = np.array([x*prob[x] for x in range(0,prob.size)])
  representation_levels = np.zeros(decision_levels.size-1, dtype=int)
  for i in range(0,representation_levels.size):
    start = int(np.ceil(decision_levels[i]))
    end = int(np.ceil(decision_levels[i+1]))
    sum_xprob = np.sum(x_prob[start:end])
    sum_prob  =np.sum(prob[start:end])
    if sum_xprob==0 or sum_prob==0:
      representation_levels[i] = (int(decision_levels[i]+decision_levels[i+1]))/2
    else:
      representation_levels[i] = sum_xprob / sum_prob
  return representation_levels

def calculate_decision_levels(repr_levels, decision_levels):
  # as we saw in the lacture:
  for i in range(1,decision_levels.size-1): # the first and last decision levels remain the same
    decision_levels[i] = (repr_levels[i-1]+repr_levels[i])/2
  return decision_levels

def calculate_MSE(hist, repr_levels, num_samples):
  mse = 0
  for x in range(0,hist.size):
    # for each x (color) in the image, find the representative by
    # subtracting x from the representatives array and find he closest
    # by taking the min difference (squarred):
    squared_error = ((repr_levels-x)**2).min()
    mse += hist[x]*squared_error
  mse /= num_samples
  return mse

def run_max_lloyd(hist, decision_levels, epsilon=0.005):
  num_of_samples = np.sum(hist)
  repr_levels=calculate_representation_levels(hist,decision_levels)
  old_mse = calculate_MSE(hist, repr_levels, num_of_samples)
  while True:
    decision_levels=calculate_decision_levels(repr_levels, decision_levels)
    repr_levels=calculate_representation_levels(hist,decision_levels)
    new_mse = calculate_MSE(hist, repr_levels, num_of_samples)
    if np.abs(new_mse - old_mse) < epsilon:
      return decision_levels, repr_levels, new_mse
    old_mse = new_mse

def create_decision_levels(image, bits):
    step = (image.max() - image.min()) /  2**bits
    decision_levels = np.arange(image.min(), image.max(), step)
    return decision_levels

img_arr = load_image_as_array("rhino.jpg")
image_hist = create_histogram(img_arr)
# show_histogram(image_hist)
bit_budgets = np.arange(1, 9)
mse_values, q_images = perform_quantization_for_bits(img_arr, bit_budgets)
show_bit_quantization_plot(bit_budgets, mse_values)

mse_values = []
arr = np.array([0, 64 , 128, 192, 256])
for b in bit_budgets:
    decision_levels = create_decision_levels(img_arr, b)
    decision_levels, repr_levels, mse = run_max_lloyd(image_hist,decision_levels,0.005)
    mse_values.append(mse)
    # ml_image = quantize_image(img_arr, decision_levels, repr_levels)

show_bit_quantization_plot(bit_budgets, mse_values)

    # new_img = ml_image.reshape(512,512)
    # old_img = img_arr.reshape(512,512)
    # Original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(old_img, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # # Quantized image
    # plt.subplot(1, 2, 2)
    # plt.imshow(new_img, cmap='gray')
    # plt.title('Quantized Image')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()