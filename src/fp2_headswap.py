# https://github.com/codersrule/REPLACE-HUMAN-HEADS-WITH-CAT-HEADS.git

# ========================================
# IMPORTS
# ========================================
import os
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from PIL import Image, ImageTk
import cv2

print(cv2.__version__)
print(cv2.__file__)


# ========================================
# CONFIGURATION CONSTANTS
# ========================================
SAVE_PATH = "/home/raoab/src/ecps205-fp2/tvi"
IMAGE_PATH = os.path.join(SAVE_PATH, "images")
VIDEO_PATH = os.path.join(SAVE_PATH, "videos")
CAT_ICON_PATH = "/home/raoab/src/ecps205-fp2/tvi/cat_icon.png"
FACE_CASCADE_PATH = "/home/raoab/fd/haarcascade_frontalface_default.xml"
TEST_VIDEO_PATH = "/home/raoab/fd/face-demographics-walking-and-pause.mp4"
WINDOW_WIDTH = 1200


# ========================================
# GLOBAL MODULE-LEVEL VARIABLES
# ========================================
no_camera = False
picam2 = None
cap = None
root = None
video_window = None
recording = False
encoder = None
video_filename = ""
countdown_label = None
image_label = None


# ========================================
# CAMERA INITIALIZATION FUNCTIONS
# ========================================
def initialize_camera():
    """Initialize camera (Picamera2 or fallback to OpenCV)."""
    global no_camera, picam2, cap
    
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration())
        print("Picamera2 detected")
        no_camera = False
    except Exception as e:
        print(f"No camera? Or Error initializing: {e}")
        no_camera = True
        cap = cv2.VideoCapture(0)


def setup_camera():
    """Configure camera settings."""
    camera_config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (640, 480)}
    )
    picam2.configure(camera_config)


def test_camera():
    """Test camera functionality."""
    picam2.start_preview(Preview.QT)
    picam2.start()
    picam2.title_fields = ["ExposureTime", "AnalogueGain"]
    time.sleep(2)
    picam2.capture_file("./test_camera_preview.jpg")


# ========================================
# HELPER FUNCTIONS
# ========================================
def get_timestamp():
    """Returns a timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(VIDEO_PATH, exist_ok=True)


# ========================================
# VIDEO WINDOW HANDLING FUNCTIONS
# ========================================
def create_video_window():
    """Creates the video feed window when preview starts."""
    global video_window, countdown_label, image_label
    
    if video_window is None or not video_window.winfo_exists():
        video_window = tk.Toplevel(root)
        video_window.title("Live Head Swap! Cam")
        video_window.protocol("WM_DELETE_WINDOW", stop_preview)
        video_window.geometry("650x550")
    
    image_label = tk.Label(video_window)
    image_label.pack()
    
    countdown_label = tk.Label(video_window, text="", font=("Arial", 30, "bold"), fg="red")
    countdown_label.pack(pady=20)


def get_frame():
    """Get frame from camera or video capture."""
    if not no_camera:
        picam2.start()
        return picam2.capture_array("main")
    else:
        ret, frame = cap.read()
        if not ret:
            return None
        return frame


def update_preview():
    """Continuously updates the Tkinter Label with live preview frames."""
    if video_window is None or not video_window.winfo_exists():
        return
    
    frame = get_frame()
    img = Image.fromarray(frame).resize((640, 480))
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img
    video_window.after(50, update_preview)


def start_preview():
    """Starts live preview and opens video feed window."""
    create_video_window()
    try:
        update_preview()
    except Exception as e:
        print(f"Error starting preview: {e}")


def stop_preview():
    """Stops live preview and closes video feed window."""
    try:
        if video_window:
            video_window.destroy()
            if not no_camera:
                picam2.stop()
    except Exception as e:
        print(f"Error stopping preview: {e}")


# ========================================
# IMAGE CAPTURE FUNCTIONS
# ========================================
def capture_image():
    """Captures an image and saves it without freezing."""
    def _capture():
        filename = os.path.join(IMAGE_PATH, f"img_{get_timestamp()}.jpg")
        try:
            print("Capturing image...")
            img = picam2.capture_image()
            img = img.convert("RGB")
            img.save(filename)
            print(f"Image saved: {filename}")
        except Exception as e:
            print(f"Error capturing image: {e}")
    
    threading.Thread(target=_capture, daemon=True).start()


# ========================================
# VIDEO RECORDING FUNCTIONS
# ========================================
def start_recording():
    """Starts the countdown timer before recording."""
    print("DEBUG: start_recording() called")
    
    if video_window is None or not video_window.winfo_exists():
        print("Please start camera first before recording!")
        messagebox.showerror("Error", "start Camera first!")
        return
    
    if countdown_label is None:
        print("ERROR: Countdown label not initialized!")
        return
    
    show_countdown_timer(3)


def show_countdown_timer(count):
    """Updates the countdown timer below the video feed."""
    print(f"DEBUG: show_countdown_timer called with count={count}")
    
    print(f"DEBUG: video_window is None? {video_window is None}")
    if video_window is not None:
        print(f"DEBUG: video_window.winfo_exists()? {video_window.winfo_exists()}")
    print(f"DEBUG: countdown_label is None? {countdown_label is None}")
    if countdown_label is not None:
        try:
            print(f"DEBUG: countdown_label.winfo_exists()? {countdown_label.winfo_exists()}")
        except:
            print("DEBUG: countdown_label.winfo_exists() threw an error")
    
    try:
        if (video_window is not None and video_window.winfo_exists() and
            countdown_label is not None and countdown_label.winfo_exists()):
            print(f"DEBUG: Setting countdown to {count}")
            countdown_label.config(text=str(count))
            if count > 0:
                video_window.after(1000, show_countdown_timer, count - 1)
            else:
                countdown_label.config(text="")
                print("DEBUG: Calling start_recording_after_countdown")
                start_recording_after_countdown()
        else:
            print("DEBUG: Condition check FAILED - countdown not displayed")
    except tk.TclError as e:
        print(f"DEBUG: TclError in countdown: {e}")
    except Exception as e:
        print(f"DEBUG: Unexpected countdown error: {e}")


def start_recording_after_countdown():
    """Starts recording video after countdown finishes."""
    print("DEBUG: start_recording_after_countdown() called")
    
    if no_camera:
        print("Recording only if camera is available")
        return
    
    global recording, encoder, video_filename
    if recording:
        print("Already recording...")
        return
    
    video_filename = os.path.join(VIDEO_PATH, f"video_{get_timestamp()}.h264")
    encoder = H264Encoder()
    try:
        print(f"Recording video: {video_filename}")
        recording = True
        picam2.stop()
        picam2.configure(picam2.create_video_configuration())
        picam2.start()
        picam2.start_recording(encoder, video_filename)
        print("DEBUG: Recording started successfully")
    except Exception as e:
        print(f"Error starting recording: {e}")


def stop_recording():
    """Stops recording and converts to MP4 without freezing."""
    global recording, encoder, video_filename
    if not recording:
        print("Not currently recording...")
        return
    
    def _stop():
        global recording, encoder, video_filename
        try:
            print("Stopping recording...")
            picam2.stop_recording()
            picam2.stop()
            picam2.configure(picam2.create_still_configuration())
            picam2.start()
            recording = False
            encoder = None
            
            if video_filename:
                mp4_filename = video_filename.replace(".h264", ".mp4")
                
                def convert_to_mp4():
                    try:
                        print(f"Converting {video_filename} to {mp4_filename}...")
                        subprocess.run(["ffmpeg", "-i", video_filename, "-c:v", "copy",
                                      "-movflags", "+faststart", mp4_filename], check=True)
                        os.remove(video_filename)
                        print(f"Conversion complete! Saved as: {mp4_filename}")
                    except Exception as e:
                        print(f"Error converting video: {e}")
                
                threading.Thread(target=convert_to_mp4, daemon=True).start()
        except Exception as e:
            print(f"Error stopping recording: {e}")
    
    threading.Thread(target=_stop, daemon=True).start()


# ========================================
# HEAD SWAP FUNCTIONS
# ========================================
def select_cat_faces():
    """Launches the cat face selection script."""
    try:
        messagebox.showerror("Error", "cat face selection not implemented!")
    except Exception as e:
        print(f"Error launching cat face selection: {e}")


def hs_settings():
    """Launches the Head Swap! settings script."""
    try:
        messagebox.showerror("Error", "hs_settings not implemented!")
    except Exception as e:
        print(f"Error launching Head Swap! settings: {e}")


def start_head_swap():
    """Starts the Head Swap! process."""
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    if face_cascade.empty():
        print("ERROR: Failed to load cascade classifier!")
        return
    
    if no_camera:
        cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nEnd of video reached. Processed {frame_count} frames.")
            break
        
        frame_count += 1
        
        if frame_count == 1:
            print(f"Frame dimensions: {frame.shape}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count == 1:
            print(f"Gray frame dimensions: {gray.shape} Gray frame dtype: {gray.dtype}")
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            print(f"Frame {frame_count}: Detected {len(faces)} face(s)")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"   Face {i+1}: x={x}, y={y}, w={w}, h={h}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: No faces detected")
        
        cv2.imshow("Head Swap! Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nUser pressed 'q' to quit")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCleanup complete")


def stop_head_swap():
    """Stops the Head Swap! process."""
    try:
        messagebox.showerror("Error", "stop_head_swap not implemented!")
    except Exception as e:
        print(f"Error stopping Head Swap!: {e}")


# ========================================
# GUI INITIALIZATION FUNCTIONS
# ========================================
def initialize_gui():
    """Initialize the main Tkinter GUI."""
    global root
    
    root = tk.Tk()
    root.title("Head Swap!")
    root.resizable(False, False)
    root.geometry(f"{WINDOW_WIDTH}x100")
    
    style = ttk.Style(root)
    style.theme_use('clam')


def create_buttons():
    """Create all GUI buttons."""
    buttons_frame = tk.Frame(root)
    buttons_frame.pack(pady=10)
    
    # Row 0 - Camera Controls
    start_preview_button = tk.Button(buttons_frame, text="Start Camera", width=18, command=start_preview)
    start_preview_button.grid(row=0, column=0, padx=5)
    
    stop_preview_button = tk.Button(buttons_frame, text="Stop Camera", width=18, command=stop_preview)
    stop_preview_button.grid(row=0, column=1, padx=5)
    
    capture_button = tk.Button(buttons_frame, text="Take a Picture", width=18, command=capture_image)
    capture_button.grid(row=0, column=2, padx=5)
    
    start_video_button = tk.Button(buttons_frame, text="Start Cam Recording", width=18, command=start_recording)
    start_video_button.grid(row=0, column=3, padx=5)
    
    stop_video_button = tk.Button(buttons_frame, text="Stop Cam Recording", width=18, command=stop_recording)
    stop_video_button.grid(row=0, column=4, padx=5)
    
    # Disable camera buttons if no camera
    if no_camera:
        start_preview_button.config(state='disable')
        start_video_button.config(state='disable')
        stop_video_button.config(state='disable')
        capture_button.config(state='disable')
        stop_preview_button.config(state='disable')
    
    # Row 1 - Head Swap Controls
    select_cat_faces_button = tk.Button(buttons_frame, text="Select Cat Faces", width=18, command=select_cat_faces)
    select_cat_faces_button.grid(row=1, column=0, padx=5)
    
    hs_settings_button = tk.Button(buttons_frame, text="Head Swap! Settings", width=18, command=hs_settings)
    hs_settings_button.grid(row=1, column=1, padx=5)
    
    # Cat Icon
    try:
        icon_image = Image.open(CAT_ICON_PATH)
        icon_image = icon_image.resize((40, 40))
        icon_photo = ImageTk.PhotoImage(icon_image)
        
        icon_label = tk.Label(buttons_frame, image=icon_photo)
        icon_label.image = icon_photo
        icon_label.grid(row=1, column=2, pady=5)
    except Exception as ex:
        print(f"Could not load icon: {ex}")
    
    start_head_swap_button = tk.Button(buttons_frame, text="Start Head Swap!", width=18, command=start_head_swap)
    start_head_swap_button.grid(row=1, column=3, padx=5)
    
    stop_head_swap_button = tk.Button(buttons_frame, text="Stop Head Swap!", width=18, command=stop_head_swap)
    stop_head_swap_button.grid(row=1, column=4, padx=5)


def on_closing():
    """Ensures clean shutdown of camera before exiting."""
    stop_preview()
    root.destroy()


def setup_window_close_handler():
    """Setup the window close handler."""
    root.protocol("WM_DELETE_WINDOW", on_closing)


# ========================================
# MAIN FUNCTION
# ========================================
def main():
    """Main entry point of the application."""
    # Step 1: Ensure directories exist
    ensure_directories()
    
    # Step 2: Initialize camera
    initialize_camera()
    
    # Step 3: Initialize GUI
    initialize_gui()
    
    # Step 4: Create buttons
    create_buttons()
    
    # Step 5: Setup window close handler
    setup_window_close_handler()
    
    # Step 6: Start Tkinter main loop
    root.mainloop()


# ========================================
# ENTRY POINT
# ========================================
if __name__ == "__main__":
    main()