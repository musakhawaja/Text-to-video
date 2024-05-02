from openai import OpenAI
from dotenv import load_dotenv
import os
from elevenlabs import clone, set_api_key, voices
import re
import requests
import json
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, ImageSequenceClip
import subprocess
import math
from PIL import Image
import numpy
import streamlit as st
import cv2

load_dotenv()
set_api_key(os.getenv("11LABS_API_KEY"))    
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))


def normalize_text(text):
    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    return text

def voice_search(pairs):
    voicess = voices()
    voices_dicts = [{"name": voice.name, "description": voice.description} for voice in voicess]
    history = {"NARRATOR": "adam"}
    audio_paths = []
    images_path = []
    text_prompts = [] 
    count = 0  
    for pair in pairs:
        person, message = pair
        print("HISTORY: ", history)
        if person not in history:
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{
                    "role": "system",
                    "content": f"""Your task is to evaluate the provided voice sample data to identify a voice that closely resembles the characteristics of {person}. Focus on matching gender, nationality, and accent as accurately as possible. You MUST ensure the voice you're choosing is not already present in the following dictionary: {history}.
                     
                     Your response must be in the following json format, only return the voice you've selected and nothing else:
                                            
                                        "person": "[YOUR ANSWER]"
                        
                        Ensure your selection is informed by the details in the data, aiming for the closest possible resemblance to the person's voice profile.
                       Please use the following voice samples: {json.dumps(voices_dicts)}"""
                }],
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            selected_voice_name = normalize_text(result["person"])
            history[person] = selected_voice_name
            print("GPT SELECTION: ", selected_voice_name)
            for voice in voicess:
                if normalize_text(voice.name) == selected_voice_name:
                    audio_path = generate_audio(voice.voice_id, message, count)
                    image_path = gen_image(person, message, count)
                    audio_paths.append(audio_path)
                    images_path.append(image_path)
                    text_prompts.append(person + ": " + message)
                
        else:
            for voice in voicess:
                if normalize_text(voice.name) == history[person]:
                    audio_path = generate_audio(voice.voice_id, message, count)
                    image_path = gen_image(person, message, count)
                    audio_paths.append(audio_path)
                    images_path.append(image_path)
                    text_prompts.append(person + ": " + message)
        count += 1
    print("AUDIO PATHS: ", audio_paths)
    return audio_paths, images_path, text_prompts

def generate_audio(voice_id, message, count):
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': os.getenv("11LABS_API_KEY"),  
        'Content-Type': 'application/json',
    }
    data = {
        "text": message,
        "voice_settings": {
            "stability": 0.30,
            "similarity_boost": 0.98,
            "style": 0.50,
            "use_speaker_boost": True
        },
        "model_id": "eleven_multilingual_v2",
    }
    response = requests.post(url, headers=headers, json=data)
    
    file_path = f"audio_{count}.mp3"
    with open(file_path, "wb") as audio_file:
        audio_file.write(response.content)
    return file_path


context = [{"role": "system", "content": """"
                You are a professional story writer. The user will ask you to write a story on a topic. Your job is to write
                a captivating story like a professional story writer. You must ensure that:
                1) the story is coherent
                2) the story makes sense
                3) the story is complete 
                4) The story is short

                While writing the story, please follow the following JSON format. It should be in a single JSON object:
                {
                    'NARRATOR': Part of the story which has no dialogues,
                    'PERSON_NAME': 'PERSON_TEXT (Dialogue that the person speaks)',
                    .....
                }
                """
             }]
history = []
def gen_image(person, text,count):
    text = f"{person} says {text}"
    print("TEXT: ", text)
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "system",
            "content": f"""Your task is to anaylyse the following text and write a prompt to generate an image from an AI image generator.
             You must follow the following instrunctions: 
             1) The prompt must be to generate characters of a story.
             2) The generated image must NOT have any text, only character of a story
             3) each picture should have 1 character but more can be present if the text allows it
             4) The prompt should be accuracte to what the text is and what the context is
             5) Analyse the previous prompts generated and create a new prompt which is relevant to the previous prompts and keeps the story coherent and visually understandable. Previous prompts: {history}
             Here is the text: {text}
                Your response should be a json object in the following format:
                
                "prompt": Your generated prompt
                
            """
        }],
        response_format={"type": "json_object"}
    )
    result = completion.choices[0].message.content
    history.append({"role": "assistant", "content": result})
    print("HISOTRY OF PROMPTs: ", history)
    data = json.loads(result)
    prompt = data["prompt"]
    print("PROMPT: ", prompt)
    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "authorization": f"Bearer {os.getenv('SD_API_KEY')}",
            "accept": "image/*"
        },
        files={"none": ''},
        data={
            "prompt": prompt,
            "output_format": "jpeg",
            "aspect_ratio": "16:9",
        },
    )


    if response.status_code == 200:
        with open(f'image_{count}.jpg', 'wb') as f:
            f.write(response.content)
        return f'image_{count}.jpg'
    else:
        print("Failed to download the image: ", response)

def reset_app():
    cleanup()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.experimental_rerun()

def extract_key_value_pairs(data):
    pattern = re.compile(r'\"(.*?)\"\s*:\s*\"(.*?)\"', re.DOTALL)
    matches = pattern.findall(data)
    return matches

def transcribe_audio(audio_path):
    with open(audio_path, 'rb') as audio:
        transcription_result = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio, 
            response_format="srt"
        )
    print(transcription_result)
    return transcription_result

def add_subtitles(video_path, srt_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"subtitles={srt_path}",
        '-codec:a', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)

def extract_audio(video_path, output_audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,          
        '-q:a', '0',               
        '-map', 'a',               
        '-y', output_audio_path     
    ]
    subprocess.run(command, check=True)

def zoom_in_effect(clip, zoom_ratio=0.008):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]

        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)

        result = numpy.array(img)
        img.close()

        return result

    return clip.fl(effect)


def adjust_image(image_path, output_width=2040, output_height=1152):
    with Image.open(image_path) as img:
        current_aspect_ratio = img.width / img.height
        target_aspect_ratio = output_width / output_height

        if current_aspect_ratio > target_aspect_ratio:
            new_width = int(target_aspect_ratio * img.height)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        elif current_aspect_ratio < target_aspect_ratio:
            new_height = int(img.width / target_aspect_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))

        img = img.resize((output_width, output_height), Image.LANCZOS)

        img.save(image_path)


def cleanup():
    for filename in os.listdir('.'):
        if filename.endswith('.mp3') or filename.endswith('.jpg'):
            os.remove(filename)
            print(f"Deleted {filename}")

def script(text):
    context.append({"role": "user", "content": text})
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=context,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    result = completion.choices[0].message.content
    return result


st.title("Video Generation App")

if 'key_value_pairs' not in st.session_state:
    st.session_state['key_value_pairs'] = []
if 'audios' not in st.session_state:
    st.session_state['audios'] = []
if 'images' not in st.session_state:
    st.session_state['images'] = []
if 'cleanup_done' not in st.session_state:
    st.session_state['cleanup_done'] = False 

# user_prompt = st.text_area("Enter your prompt to generate a video:", height=150)
# generate_button = st.button("Generate Script")

# if generate_button:
#     with st.spinner('Processing your script...'):
#         result = script(user_prompt)
#         st.session_state['edited_script'] = result  

choice = st.radio("Do you want to generate a new script or enter your own?", ('Generate', 'Enter'))

if choice == 'Generate':
    user_prompt = st.text_area("Enter your prompt to generate a video:", height=150)
    generate_button = st.button("Generate Script")

    if generate_button:
        with st.spinner('Processing your script...'):
            result = script(user_prompt)  # Call your script function here
            st.session_state['edited_script'] = result  

elif choice == 'Enter':
    # Provide a text area for the user to write or paste their script
    manual_script = st.text_area("Enter your script here:", height=300)
    submit_script_button = st.button("Submit Script")

    if submit_script_button:
        st.session_state['edited_script'] = manual_script 

if 'edited_script' in st.session_state:
    edited_script = st.text_area("Edit the script as necessary:", value=st.session_state['edited_script'], height=300)
    confirm_button = st.button("Confirm Script Edits")

    if confirm_button:
        with st.spinner('Updating script and generating audio/images...'):
            key_value_pairs = extract_key_value_pairs(edited_script)
            st.session_state['key_value_pairs'] = key_value_pairs

            audios, images, text_prompts = voice_search(key_value_pairs)
            st.session_state['audios'] = audios
            st.session_state['images'] = images
            st.session_state["text_prompts"] = text_prompts

if st.session_state['images']:
    for idx, (image, prompt) in enumerate(zip(st.session_state['images'], st.session_state['text_prompts'])):
        try:
            st.image(image, caption=f"Image {idx + 1}: {prompt}", use_column_width=True)
        except Exception as e:
            st.session_state['images'][idx] = None  # Set the problematic image to None
            st.write(f"Placeholder for Image {idx + 1}: {prompt} (image not available)")
            print(f"Failed to load image {idx + 1}: {prompt}. Error: {e}")

        
    image_to_replace = st.selectbox("Select an image to replace:", options=list(range(len(st.session_state['images']))), format_func=lambda x: f"Image {x + 1}")
    new_image = st.file_uploader("Upload new image:", type=["jpg", "jpeg", "png"], key="new_image_uploader")
    replace_button = st.button("Replace Image")

    # if replace_button and new_image:
    #     image_path = st.session_state['images'][image_to_replace]
    #     with open(image_path, "wb") as f:
    #         f.write(new_image.getvalue())
    #     st.success(f"Replaced Image {image_to_replace + 1}")

    if replace_button and new_image:
        if st.session_state['images'][image_to_replace] is None or os.path.exists(st.session_state['images'][image_to_replace]):
            image_path = f'image_{image_to_replace}.jpg'  # Define a standard path for new uploads
            with open(image_path, "wb") as f:
                f.write(new_image.getvalue())
            adjust_image(image_path)
            st.session_state['images'][image_to_replace] = image_path
            st.success(f"Replaced Image {image_to_replace + 1}")

    if st.button("Generate Video"):
        with st.spinner('Generating video and subtitles...'):
            clips = []
            for audio, image in zip(st.session_state['audios'], st.session_state['images']):
                try:
                    audio_clip = AudioFileClip(audio)
                    img_clip = ImageClip(image, duration=audio_clip.duration).set_audio(audio_clip)
                    img_clip = zoom_in_effect(img_clip, 0.02)
                    clips.append(img_clip)
                except Exception as e:
                    continue
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip_path = "output_video.mp4"
            final_clip.write_videofile(final_clip_path, codec="libx264", fps=30)
            extract_audio("output_video.mp4", "final_audio.mp3")
            transcription_result = transcribe_audio("final_audio.mp3")
            cleanup()
            srt_path = "final_subtitles.srt"
            with open(srt_path, 'w') as file:
                file.write(transcription_result)

            add_subtitles(final_clip_path, srt_path, "final_video_with_subtitles.mp4")

            
            if os.path.exists("final_video_with_subtitles.mp4"):
                video_file_path = "final_video_with_subtitles.mp4"
                st.video(video_file_path)
                st.download_button(
                    label="Download Video",
                    data=open(video_file_path, "rb"),
                    file_name="final_video_with_subtitles.mp4",
                    mime="video/mp4"
                )
                os.remove(video_file_path)
            else:
                st.error("Failed to generate video.")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
