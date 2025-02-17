# main.py

import os
import asyncio
import cv2
import base64
import numpy as np

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai

from dotenv import load_dotenv

# Import the color sensor helper function
from color_sensor import get_color_sensor_data

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def capture_camera_frame():
    """Capture a frame from the camera and return it as a base64-encoded string."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture frame.")
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


async def entrypoint(ctx: JobContext):
    # Read the current mucus color from the sensor
    color_data = get_color_sensor_data()
    if "error" not in color_data:
        mucus_color_context = (
            f"Current mucus color is {color_data['hex']} "
            f"(R: {color_data['r']}, G: {color_data['g']}, B: {color_data['b']})."
        )
    else:
        mucus_color_context = "Mucus color sensor error: " + color_data["error"]

    # Initialize the conversation context with your agent, including the mucus color
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You're a warm and friendly doctor voice assistant, specializing in sinus health. "
                    "Speak naturally, like a caring friend who is also an expert. Keep things light, supportive, "
                    "and easy to understand—no complex medical jargon. Pretend we're having a human conversation, "
                    "no special formatting or headings, just natural speech. "
                    "You can also see the user's camera feed and respond to visual queries.\n\n"
                    f"Note: The current mucus color is as follows: {mucus_color_context}"
                ),
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model="meta-llama/llama-3-70b-instruct",
        ),
        tts=cartesia.TTS(voice="248be419-c632-4f23-adf1-5324ed7dbf1d"),
        chat_ctx=initial_ctx,
    )

    await ctx.connect()
    assistant.start(ctx.room)

    # Start with a greeting
    await assistant.say("Hi there, how are you doing today?", allow_interruptions=True)

    # Continuous conversation loop
    while True:
        try:
            # Listen for user input
            user_input = await assistant.listen()
            if user_input:
                # Capture camera frame if the user asks a visual question
                if "see" in user_input.lower() or "look" in user_input.lower():
                    frame_base64 = await capture_camera_frame()
                    visual_context = (
                        f"Here is the current camera feed: "
                        f"![frame](data:image/jpeg;base64,{frame_base64})"
                    )
                    user_input = f"{user_input}\n\n{visual_context}"

                # Update mucus color context (in case it changes during conversation)
                color_data = get_color_sensor_data()
                if "error" not in color_data:
                    mucus_color_context = (
                        f"Current mucus color is {color_data['hex']} "
                        f"(R: {color_data['r']}, G: {color_data['g']}, B: {color_data['b']})."
                    )
                else:
                    mucus_color_context = "Mucus color sensor error: " + color_data["error"]

                # Append the mucus color info to the user's input context
                user_input = f"{user_input}\n\n{mucus_color_context}"

                # Generate a response using the LLM
                response = await assistant.llm.chat(
                    messages=[ChatMessage(role="user", content=user_input)]
                )
                # Speak the response
                await assistant.say(response, allow_interruptions=True)
        except Exception as e:
            print(f"Error in conversation loop: {e}")
            break


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
