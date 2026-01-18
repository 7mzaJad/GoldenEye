from inference import predict_image

result = predict_image("test_image.jpg")
print("Person detected" if result == 1 else "No person detected")
