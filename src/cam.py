#2025/11/15 ECPS205 Final Project... KEVIN_LEE
from picamera2 import Picamera2, Preview
import time as time
import os
import numpy as np

class CameraController:
    def __init__(self, camid = 0, preview = False):
        """Init Camera"""
        self.camid = camid
        self.picam = Picamera2(camid)
        self.preview = preview
        self.imgid = 0
        self.shot_arrays = []
        print("Camera Init\n")
    
    def start_preview(self):
        
        if not self.preview:
            # Try different preview modes based on environment
            display = os.environ.get('DISPLAY', '')
            
            try:
                if display and 'localhost' in display:
                    # Remote X11 - use QT
                    print("Using QT preview for X11 forwarding...")
                    self.picam.start_preview(Preview.QT)
                elif display:
                    # Local display - try QTGL first
                    print("Using QTGL preview...")
                    self.picam.start_preview(Preview.QTGL)
                else:
                    # No display - run headless
                    print("No display detected, running headless...")
            except Exception as e:
                print(f"Preview failed: {e}")
                print("Running in headless mode...")
            
            self.picam.start()
            time.sleep(1)
            self.preview = True
            print("Camera started\n")
            
    def stop_preview(self):
        
        if self.preview:
            try:
                self.picam.stop_preview()
            except:
                pass
            self.picam.stop()
            self.preview = False
            print("Camera stopped\n")
            
    def take_shot(self):
        
        self.start_preview()
        take = input(r"[S] to take a shot >o<...[Q] to end App...[L] to list shots...")
        take = take.lower()
        
        if take == "s":
            # Capture as array
            img_array = self.picam.capture_array("main")
            self.shot_arrays.append(img_array)
            
            # Save as jpg for viewing image purposes
            filename = f"cam0_{self.imgid}.jpg"
            self.picam.capture_file(filename)
            
            print(f"Shot {self.imgid} captured")
            print(f"File: {filename}")
            print(f"Array: shape={img_array.shape}, dtype={img_array.dtype}")
            print(f"Memory: {img_array.nbytes/1024:.1f} KB")
            print(f"Total shots in memory: {len(self.shot_arrays)}")
            self.imgid += 1
            
        elif take == "l":
            self.list_shots()
            
        elif take == "q":
            print("Exiting application...")
        else:
            print("Invalid input. Use S/Q/L")
            
        return take
    
    def list_shots(self):
        """List all captured shots"""
        if len(self.shot_arrays) == 0:
            print("\nNo shots captured yet.\n")
        else:
            print(f"\nCaptured Shots ({len(self.shot_arrays)}) ")
    
    def get_shot(self, index):
        """Get a specific shot by index"""
        if 0 <= index < len(self.shot_arrays):
            return self.shot_arrays[index]
        else:
            print(f"Error: Shot index {index} out of range (0-{len(self.shot_arrays)-1})")
            return None
    
    def get_all_shots(self):
        """Get all captured shots as a list"""
        return self.shot_arrays
    
    def get_shots_as_array(self):
        """Get all shots stacked as a single 4D numpy array"""
        if len(self.shot_arrays) > 0:
            return np.array(self.shot_arrays)
        return None
    
    def clear_shots(self):
        """Clear all shots from memory"""
        count = len(self.shot_arrays)
        self.shot_arrays.clear()
        print(f"Cleared {count} shots from memory.")
    
    def save_shots_to_file(self, filename="shots_data.npz"):
        """Save all shot arrays to a compressed numpy file"""
        if len(self.shot_arrays) > 0:
            # Save with metadata
            np.savez_compressed(
                filename,
                images=np.array(self.shot_arrays),
                count=len(self.shot_arrays)
            )
            print(f"Saved {len(self.shot_arrays)} shots to {filename}")
        else:
            print("No shots to save.")
    
    def load_shots_from_file(self, filename="shots_data.npz"):
        """Load shot arrays from a numpy file"""
        try:
            data = np.load(filename)
            self.shot_arrays = list(data['images'])
            print(f"Loaded {len(self.shot_arrays)} shots from {filename}")
            self.imgid = len(self.shot_arrays)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
        except Exception as e:
            print(f"Error loading file: {e}")
