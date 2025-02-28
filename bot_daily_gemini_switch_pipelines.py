#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    InputAudioRawFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import LLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.services.google.google import GoogleLLMContext
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

system_message = None


class UserAudioCollector(FrameProcessor):
    """This FrameProcessor collects audio frames in a buffer, then adds them to the
    LLM context when the user stops speaking.
    """

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
                # frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


async def respond_with_apple(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    """Function the bot can call to return Apple."""

    await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


async def respond_with_banana(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    """Function the bot can call to return Banana."""

    await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


async def respond_with_orange(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    """Function the bot can call to return Orange."""

    await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


async def terminate_call(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    """Function the bot can call to terminate the call upon completion of the call."""

    await llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


async def main(
    room_url: str,
    token: str,
    callId: str,
    callDomain: str,
    detect_voicemail: bool,
    dialout_number: Optional[str],
):
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            # transcription_enabled=True,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    ### APPLE PIPELINE

    tools = [
        {
            "function_declarations": [
                {
                    "name": "respond_with_apple",
                    "description": "Call this function when the user asks about apples.",
                },
                {
                    "name": "respond_with_banana",
                    "description": "Call this function when the user asks about bananas.",
                },
                {
                    "name": "respond_with_orange",
                    "description": "Call this function when the user asks about oranges.",
                },
                {
                    "name": "terminate_call",
                    "description": "Call this function to terminate the call.",
                },
            ]
        }
    ]

    respond_with_apple_instruction = """
Always respond with the word 'Apple'.

You have access to the following functions that you can call:
- respond_with_apple: Call this function when the user asks about apples.
- respond_with_banana: Call this function when the user asks about bananas.
- respond_with_orange: Call this function when the user asks about oranges.
- terminate_call: Call this function to terminate the call.

If the user mentions bananas or oranges, call the appropriate function instead of responding with 'Apple'.
If the user asks to terminate the call, call the terminate_call function.
"""

    apple_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=respond_with_apple_instruction,
        tools=tools,
    )

    apple_context = GoogleLLMContext()
    apple_context_aggregator = apple_llm.create_context_aggregator(apple_context)

    apple_llm.register_function("respond_with_apple", respond_with_apple)
    apple_llm.register_function("respond_with_banana", respond_with_banana)
    apple_llm.register_function("respond_with_orange", respond_with_orange)
    apple_llm.register_function("terminate_call", terminate_call)

    apple_audio_collector = UserAudioCollector(apple_context, apple_context_aggregator.user())

    apple_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            apple_audio_collector,  # Collect audio frames
            apple_context_aggregator.user(),  # User responses
            apple_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            apple_context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )
    apple_pipeline_task = PipelineTask(
        apple_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])

    runner = PipelineRunner()

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await apple_pipeline_task.queue_frame(EndFrame())

    print("!!! starting apple pipeline")
    await runner.run(apple_pipeline_task)
    print("!!! Done with apple pipeline")

    ### BANANA PIPELINE

    respond_with_banana_instruction = """
    Always respond with the word 'Banana'.

    You have access to the following functions that you can call:
    - respond_with_apple: Call this function when the user asks about apples.
    - respond_with_banana: Call this function when the user asks about bananas.
    - respond_with_orange: Call this function when the user asks about oranges.
    - terminate_call: Call this function to terminate the call.

    If the user mentions apples or oranges, call the appropriate function instead of responding with 'Banana'.
    If the user asks to terminate the call, call the terminate_call function.
    """

    banana_llm = GoogleLLMService(
        # model="models/gemini-2.0-flash-lite",
        model="models/gemini-2.0-flash-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=respond_with_banana_instruction,
        tools=tools,
    )
    banana_context = GoogleLLMContext()

    banana_context_aggregator = banana_llm.create_context_aggregator(banana_context)

    banana_llm.register_function("respond_with_apple", respond_with_apple)
    banana_llm.register_function("respond_with_banana", respond_with_banana)
    banana_llm.register_function("respond_with_orange", respond_with_orange)
    banana_llm.register_function("terminate_call", terminate_call)

    # banana_audio_collector = UserAudioCollector(banana_context, banana_context_aggregator.user())

    banana_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            banana_context_aggregator.user(),  # User responses
            banana_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            banana_context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    banana_pipeline_task = PipelineTask(
        banana_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await apple_pipeline_task.queue_frame(EndFrame())
        await banana_pipeline_task.queue_frame(EndFrame())

    print("!!! starting banana pipeline")
    banana_context_aggregator.user().set_messages(
        [
            {
                "role": "system",
                "content": respond_with_banana_instruction,
            }
        ]
    )
    await banana_pipeline_task.queue_frames([banana_context_aggregator.user().get_context_frame()])
    await runner.run(banana_pipeline_task)

    print("!!! Done with banana pipeline")

    ### ORANGE PIPELINE

    respond_with_orange_instruction = """
    Always respond with the word 'Orange'.

    You have access to the following functions that you can call:
    - respond_with_apple: Call this function when the user asks about apples.
    - respond_with_banana: Call this function when the user asks about bananas.
    - respond_with_orange: Call this function when the user asks about oranges.
    - terminate_call: Call this function to terminate the call.

    If the user mentions apples or bananas, call the appropriate function instead of responding with 'Orange'.
    If the user asks to terminate the call, call the terminate_call function.
    """

    orange_llm = GoogleLLMService(
        # model="models/gemini-2.0-flash-lite",
        model="models/gemini-2.0-flash-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=respond_with_orange_instruction,
        tools=tools,
    )

    orange_context = GoogleLLMContext()

    orange_context_aggregator = orange_llm.create_context_aggregator(orange_context)

    # orange_audio_collector = UserAudioCollector(orange_context, orange_context_aggregator.user())

    orange_llm.register_function("respond_with_apple", respond_with_apple)
    orange_llm.register_function("respond_with_banana", respond_with_banana)
    orange_llm.register_function("respond_with_orange", respond_with_orange)
    orange_llm.register_function("terminate_call", terminate_call)

    orange_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            orange_context_aggregator.user(),  # User responses
            orange_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            orange_context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    orange_pipeline_task = PipelineTask(
        orange_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await apple_pipeline_task.queue_frame(EndFrame())
        await banana_pipeline_task.queue_frame(EndFrame())
        await orange_pipeline_task.queue_frame(EndFrame())

    print("!!! starting orange pipeline")
    orange_context_aggregator.user().set_messages(
        [
            {
                "role": "system",
                "content": respond_with_orange_instruction,
            }
        ]
    )
    await orange_pipeline_task.queue_frames([orange_context_aggregator.user().get_context_frame()])
    await runner.run(orange_pipeline_task)

    print("!!! Done with orange pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Simple ChatBot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-d", type=str, help="Call Domain")
    parser.add_argument("-v", action="store_true", help="Detect voicemail")
    parser.add_argument("-o", type=str, help="Dialout number", default=None)
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.d, config.v, config.o))
