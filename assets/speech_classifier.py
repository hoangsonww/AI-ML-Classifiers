from moviepy.editor import VideoFileClip
import cv2
import pyaudio
import speech_recognition as sr
import numpy as np
import threading


def monitor_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    print("Please speak now. Monitoring audio...")

    try:
        while True:
            data = np.frombuffer(stream.read(1024), dtype=np.int16)
            volume = np.abs(data).mean()
            print(f"Volume: {volume:.2f}", end='\r')
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()


# NOTE: Before running this code, ENSURE that your microphone is connected and working properly. Go to settings and check if
# the microphone is enabled, connected, selected as the input, and the volume is set to an appropriate level.
# Normal volume level detected by our script is around 70-100%. If the volume level is below 70%, the script may not
# detect your speech properly.
def recognize_speech_from_mic(recognizer):
    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Please speak now. Listening...")
        audio = recognizer.listen(source)

    print("\nProcessing...")
    try:
        transcription = recognizer.recognize_google(audio)
        print("You said: " + transcription)
        return transcription
    except sr.RequestError:
        print("API unavailable. Please check your internet connection.")
    except sr.UnknownValueError:
        print("Unable to recognize speech.")


# NOTE: This functionality will process the audio in chunks of 15 seconds each. You can adjust the duration of each chunk
# by changing the value of the 'audio_chunk_duration' variable.
def process_audio_chunk(recognizer, audio_path, start_time, duration):
    with sr.AudioFile(audio_path) as source:
        recognizer.record(source, duration=start_time)  # skip to start of chunk
        audio_chunk = recognizer.record(source, duration=duration)
    try:
        print("Recognizing speech from video chunk...")
        chunk_transcription = recognizer.recognize_google(audio_chunk)
        print(f"Chunk transcription: {chunk_transcription}")
        return chunk_transcription
    except sr.UnknownValueError:
        print("Could not understand audio or audio is currently silent. Stay tuned...")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    return ""


def annotate_video(video_path, recognizer):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "16000"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    audio_thread = None
    audio_chunk_duration = 15  # seconds
    audio_chunk_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % int(fps * audio_chunk_duration) == 0:
            if audio_thread and audio_thread.is_alive():
                audio_thread.join()
            audio_thread = threading.Thread(target=process_audio_chunk,
                                            args=(recognizer, audio_path, audio_chunk_start, audio_chunk_duration))
            audio_thread.start()
            audio_chunk_start += audio_chunk_duration

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key & 0xFF in [ord('q'), 27]:  # 27 is the ESC key
            break
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break

    if audio_thread and audio_thread.is_alive():
        audio_thread.join()

    cap.release()
    cv2.destroyAllWindows()


def main():
    recognizer = sr.Recognizer()
    choice = input("Choose 'microphone' or 'video': ").lower()

    if choice == 'microphone':
        monitor_audio_thread = threading.Thread(target=monitor_audio)
        monitor_audio_thread.start()

        transcription = recognize_speech_from_mic(recognizer)
        if transcription:
            print("Transcription from microphone:", transcription)
        else:
            print("No transcription available.")
    elif choice == 'video':
        video_path = input("Enter the video file path: ")
        annotate_video(video_path, recognizer)


if __name__ == "__main__":
    main()
