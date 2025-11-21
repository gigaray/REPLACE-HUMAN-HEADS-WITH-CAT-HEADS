#2025/11/15 ECPS205 Final Project... KEVIN_LEE
from cam import CameraController
import time as time

def main():
    print("Commands: [S]: Take shot, [Q]: Quit, [L]: List files\n")
    
    cam0 = CameraController(0, False)
    cam0.start_preview()
    
    try:
        while True:
            cmd = cam0.take_shot()
            
            if cmd == "q":
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
