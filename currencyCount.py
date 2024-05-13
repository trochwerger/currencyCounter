from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time

# Using custom YOLO model trained on custom dataset
model = YOLO("runs/detect/train3/weights/best.pt")

sample_image = "sample_image.jpg"

# Initialize dictionary to store count of each bill
bill_counts_image = {"5 cad": 0, "10 cad": 0 ,"20 cad": 0, "50 cad": 0, "100 cad": 0}


# print("Enter (1) for Folder, (2) for image: ")
choice = int(input("Enter (1) for Image, (2) for Folder, (3) for Live Video: "))

# Single image detection
if choice == 1:
    while True:
        file = input("Enter file path or 0 for default, or q to quit: ")
        if file == "q":
            exit()
        elif file == "0":
            file = sample_image
        if Path(file).exists():
            break
        else:
            print("File does not exist. Please enter a valid path.")
        
    img_file = cv2.imread(file)

    # Using custom model to detect objects in the image, with a confidence threshold of 0.8, stream = False because we are using a single image
    results = model(img_file, stream=False, conf= 0.8)  

    for result in results:
        # Create a copy of the original image
        img = np.copy(result.orig_img)
        # Plot the detection results on the image
        img = result.plot()
        print(result)

        current_sum = 0

        for ci, c in enumerate(result):
            # Get the label of the detected object
            label = c.names[c.boxes.cls.tolist().pop()]
            # Split the label to get the cash value
            cash_value = label.split(" ")[0]
            # Add the cash value to the current sum
            current_sum += int(cash_value)
            # Increment the count for the detected bill
            bill_counts_image[label] += 1

         # Create a new blank white image
        receipt = np.ones((500, 500, 3), dtype=np.uint8) * 255

        # Draw the total sum on the receipt
        receipt = cv2.putText(receipt, f"Total sum: {current_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw the count for each bill on the receipt
        y_offset = 60
        for bill, count in bill_counts_image.items():
            receipt = cv2.putText(receipt, f"{bill}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30  # Move the y-coordinate for the next line
        
        # Display the receipt
        cv2.imshow("Receipt", receipt)
        cv2.imshow("Annotated Frame", img)
        cv2.waitKey(0)

        # print("Current sum", current_sum)

        img = cv2.putText(img, f"Total sum: {current_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        run_folder = f"./results/result_run_{int(time.time())}"
        os.makedirs(run_folder, exist_ok=True)
        
        # Save the result image to the new directory
        cv2.imwrite(f'{run_folder}/result.jpg', img)

        # Save receipt to the new directory
        cv2.imwrite(f'{run_folder}/receipt.jpg', receipt)

elif choice == 2:
    run_folder = f"results/result_run_{int(time.time())}"
    os.makedirs(run_folder, exist_ok=True)

    while True:
        folder_path = input("Enter folder path or 0 for default, or q to quit: ")
        if folder_path == "q":
            exit()
        elif folder_path == "0":
            folder_path = "sample_images"
        if Path(folder_path).exists():
            break
        else:
            print("Folder does not exist. Please enter a valid path.")

    image_files = os.listdir(folder_path)
    image_files = [file for file in image_files if file.endswith((".jpg", ".jpeg", ".png"))]

    # Initialize variable to store sum of all cash in images in folder
    total_sum = 0
    current_sum = 0
    bill_counts_total = {"5 cad": 0, "10 cad": 0, "20 cad": 0, "50 cad": 0, "100 cad": 0}

    for file_index, file_path in enumerate(image_files):
        # Using custom model to detect objects in the image, with a confidence threshold of 0.8, stream = True because we are using multiple images and we want to avoid Out of Memory errors by reducing intermediate tensor storage
        results = model([os.path.join(folder_path, file_path)], stream=True)  # return a list of Results objects

        for result in results:
            # Create a copy of the original image
            img = np.copy(result.orig_img)
            # plot(): Plots detection results on an input image, returning an annotated image.
            img = result.plot()

            current_sum = 0
            bill_count_current = {"5 cad": 0, "10 cad":0, "20 cad": 0, "50 cad": 0, "100 cad": 0}

            for ci,c in enumerate(result):
                # Get the label of the detected object
                label = c.names[c.boxes.cls.tolist().pop()]
                # Split the label to get the cash value
                cash_value = label.split(" ")[0]
                # Add the cash value to the current sum
                current_sum += int(cash_value)
                # Increment total cash in all image files in folder
                total_sum += int(cash_value)
                # Increment the count for the detected bill
                bill_count_current[label] += 1
                # Increment the count for the detected bill in all image files in folder
                bill_counts_total[label] += 1

            # Create a new blank image with additional space at the top
            border_size = 300  # Adjust this value as needed
            new_img = np.zeros((img.shape[0] + border_size, img.shape[1], img.shape[2]), dtype=np.uint8)
            # Copy the original image onto the new image
            new_img[border_size:] = img

            # Draw the total sum on the new image
            new_img = cv2.putText(new_img, f"Total sum: {current_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw the count for each bill on the new image
            y_offset = 60
            for bill, count in bill_count_current.items():
                new_img = cv2.putText(new_img, f"{bill}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 30  # Move the y-coordinate for the next line

            # Save the result image to the new directory
            cv2.imwrite(f'{run_folder}/{file_path}_result.jpg', new_img)
            cv2.imshow("Annotated Frame", new_img)
            cv2.waitKey(0)
        receipt_total = np.ones((500, 500, 3), dtype=np.uint8) * 255
        receipt_total = cv2.putText(receipt_total, f"Total sum: {total_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        y_offset = 60
        for bill, count in bill_counts_total.items():
            receipt_total = cv2.putText(receipt_total, f"{bill}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 30

        # Display the total receipt
    cv2.imshow("Total Receipt", receipt_total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif choice == 3:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=False, conf=0.8)

        for result in results:
            img = np.copy(result.orig_img)
            img = result.plot()

            current_sum = 0

            for ci, c in enumerate(result):
                label = c.names[c.boxes.cls.tolist().pop()]
                cash_value = label.split(" ")[0]
                current_sum += int(cash_value)
                bill_counts_image[label] += 1

            img = cv2.putText(img, f"Total sum: {current_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            run_folder = f"./results/result_run_{int(time.time())}"
            os.makedirs(run_folder, exist_ok=True)

            cv2.imwrite(f'{run_folder}/result.jpg', img)

            cv2.imshow("Annotated Frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

