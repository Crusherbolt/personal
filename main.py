import os
import asyncio
import cv2
import base64
import numpy as np

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai

from dotenv import load_dotenv

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
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You're a warm and friendly doctor voice assistant, specializing in sinus health. Speak naturally, like a caring friend who also happens to be an expert. Keep things light, supportive, and easy to understand—no complex medical jargon. Pretend we're having a human conversation, no special formatting or headings, just natural speech. You can also see the user's camera feed and respond to visual queries.",
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
                    visual_context = f"Here is the current camera feed: ![frame](data:image/jpeg;base64,{frame_base64})"
                    user_input = f"{user_input}\n\n{visual_context}"

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