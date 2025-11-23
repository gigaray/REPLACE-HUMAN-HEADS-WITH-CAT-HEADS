#2025/11/15 ECPS205 Final Project... KEVIN_LEE
from cam import CameraController
import time as time
from cat import make_cat, remove_white_bg, cat_paste
import cv2 as cv
import numpy as np

POS_X = 200
POS_Y = 150

def main():
    print("Commands: [S]: Take shot, [Q]: Quit, [L]: List files\n")
    
    cam0 = CameraController(0, False)
    #cam0.start_preview()
    
    # ---- YOU control these ----
    POS_X = 200   # change this later
    POS_Y = 150   # change this later
    # --------------------------

    sprite = cv.imread("cat.png", cv.IMREAD_UNCHANGED)
    corp_cat = make_cat(sprite, 5)      # choose which cat
    cat = remove_white_bg(corp_cat)     # make background transparent

    # Resize if you want
    cat = cv.resize(cat, (120, 120))

    try:
        while True:
            # Capture live frame
            frame = cam0.picam.capture_array("main")

            # Paste cat on the frame
            frame = cat_paste(frame, cat, POS_X, POS_Y)

            # Show result
            cv.imshow("Camera with Cat", frame)

            # Check for key press (q quits)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    print("\nStopping camera application....")
    
    # Save arrays if any were captured
    if len(cam0.get_all_shots()) > 0:
        save = input(f"\nSave {len(cam0.get_all_shots())} captured arrays to .npz file? [Y/N]: ").lower()
        if save == "y":
            filename = input("Enter filename (default: shots_data.npz): ").strip()
            if not filename:
                filename = "shots_data.npz"
            if not filename.endswith('.npz'):
                filename += '.npz'
            cam0.save_shots_to_file(filename)
    
    time.sleep(1)
    cam0.stop_preview()
    
    print(f"Session complete. Total shots captured: {len(cam0.get_all_shots())}")


if __name__ == "__main__":
    main()
