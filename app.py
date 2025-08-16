import streamlit as st
import os
from tracker import run_tracker_live

st.set_page_config(page_title="Object Tracker", layout="centered")
st.title("Real-Time Object Tracker")
st.write("Track an object in a video or through your webcam.")

# Tracking mode
mode = st.radio("Choose mode:", ["Upload a Video", "Live Camera"])

# if mode == "📹 Upload a Video":
#     video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

#     if video_file is not None:
#         video_path = os.path.join("uploaded_video.mp4")
#         with open(video_path, "wb") as f:
#             f.write(video_file.read())

#         st.success("Video uploaded successfully!")

#         if st.button("Start Video Tracking"):
#             st.info("Select the object in the opened window.")
#             run_tracker_video(video_path)

if mode == "Live Camera":
    if st.button("Start Live Camera Tracking"):
        # st.info("Select the object in the first frame.")
        run_tracker_live()
