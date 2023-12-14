import streamlit as st
from octoai.client import Client
from octoai.errors import OctoAIClientError, OctoAIServerError
from octoai.types import Video
from io import BytesIO
from base64 import b64encode
from PIL import Image, ExifTags
import os
from tempfile import NamedTemporaryFile
import time

SVD_ENDPOINT_URL = os.environ["SVD_ENDPOINT_URL"]
OCTOAI_TOKEN = os.environ["OCTOAI_TOKEN"]

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

def rotate_image(image):
    try:
        # Rotate based on Exif Data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image
    except:
        return image

def img2vid(my_upload, num_videos=2):
    # Wrap all of this in a try block
    col1, col2 = st.columns(2)
    try:
        # OctoAI client
        oai_client = Client(OCTOAI_TOKEN)

        progress_text = "Video generation in action on OctoAI..."
        percent_complete = 0
        progress_bar = st.progress(percent_complete, text=progress_text)

        # Rotate image and perform some rescaling
        input_img = Image.open(my_upload)
        input_img = rotate_image(input_img)

        col1.image(input_img)

        futures = []
        for vid in range(0, num_videos):
            inputs = {
                "input": {
                    "image": read_image(input_img),
                }
            }
            future = oai_client.infer_async(endpoint_url=f"{SVD_ENDPOINT_URL}/infer", inputs=inputs)
            futures.append(future)

        for future in futures:
            while not oai_client.is_future_ready(future):
                time.sleep(0.5)
                percent_complete = min(99, percent_complete+1)
                if percent_complete == 99:
                    progress_text = "Video generation is taking longer than usual, hang tight!"
                progress_bar.progress(percent_complete, text=progress_text)
            result = oai_client.get_future_result(future)
            video = Video.from_endpoint_response(result, key="output")
            progress_bar.empty()
            f = NamedTemporaryFile()
            video.to_file(f.name)
            video_file = open(f.name, 'rb')
            video_bytes = video_file.read()
            col2.video(video_bytes)

    except OctoAIClientError as e:
        st.write("Oops something went wrong (client error)!")
        progress_bar.empty()

    except OctoAIServerError as e:
        st.write("Oops something went wrong (server error)")
        progress_bar.empty()

    except Exception as e:
        st.write("Oops something went wrong (unexpected error)!")
        progress_bar.empty()


st.set_page_config(layout="wide", page_title="Img2Vid - Powered by OctoAI")

st.write("## Image2Video - Powered by OctoAI")
st.write("\n\n")
st.write("### For OctoML internal use only!")


my_upload = st.file_uploader("Take a snap or upload a photo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    img2vid(my_upload)
