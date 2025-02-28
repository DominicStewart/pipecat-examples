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
from typing import Optional, Dict, List

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

# Global configs
daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


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


class DynamicPipelineManager:
    """Manages multiple pipelines that can be switched between."""

    def __init__(self):
        self.current_pipeline = None
        self.next_pipeline = None
        self.transport = None
        self.tts = None
        self.stt = None
        self.runner = None
        self.dialout_number = None
        self.detect_voicemail = False
        self.should_speak_first = False
        # We only need functions to switch from apple to banana/orange
        self.fruit_functions = {
            "respond_with_banana": self.switch_to_banana,
            "respond_with_orange": self.switch_to_orange,
        }

    def initialize(self, transport, tts, stt, runner, dialout_number=None, detect_voicemail=False):
        """Initialize the pipeline manager with transport and services."""
        self.transport = transport
        self.tts = tts
        self.stt = stt
        self.runner = runner
        self.dialout_number = dialout_number
        self.detect_voicemail = detect_voicemail

        # Set up appropriate handlers based on the call mode
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handlers based on dial mode."""
        if self.dialout_number:
            logger.debug(f"Dialout number detected; doing dialout to: {self.dialout_number}")

            # Configure handlers for dialing out
            @self.transport.event_handler("on_joined")
            async def on_joined(transport, data):
                logger.debug(f"Joined; starting dialout to: {self.dialout_number}")
                await transport.start_dialout({"phoneNumber": self.dialout_number})

            @self.transport.event_handler("on_dialout_connected")
            async def on_dialout_connected(transport, data):
                logger.debug(f"Dial-out connected: {data}")

            @self.transport.event_handler("on_dialout_answered")
            async def on_dialout_answered(transport, data):
                logger.debug(f"Dial-out answered: {data}")

            @self.transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"First participant joined (dialout mode): {participant}")
                # Always enable transcription for when we need it in later pipelines
                try:
                    await transport.capture_participant_transcription(participant["id"])
                    logger.info(f"Capturing transcription for participant: {participant['id']}")
                except Exception as e:
                    logger.error(f"Error capturing transcription: {e}")
                # In dialout, the caller speaks first, so we don't queue a context frame
                logger.info("Dialout mode: waiting for user to speak first")

        elif self.detect_voicemail:
            logger.debug("Detect voicemail mode enabled")

            @self.transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"First participant joined (voicemail detection mode): {participant}")
                try:
                    await transport.capture_participant_transcription(participant["id"])
                    logger.info(f"Capturing transcription for participant: {participant['id']}")
                except Exception as e:
                    logger.error(f"Error capturing transcription: {e}")
                # In voicemail detection, we wait for the voicemail greeting
                logger.info("Voicemail detection mode: waiting for voicemail greeting")

        else:
            logger.debug("No dialout number; assuming dialin")

            @self.transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"First participant joined (dialin mode): {participant}")
                try:
                    await transport.capture_participant_transcription(participant["id"])
                    logger.info(f"Capturing transcription for participant: {participant['id']}")
                except Exception as e:
                    logger.error(f"Error capturing transcription: {e}")
                # For dialin, we'll set a flag to make the bot speak first
                self.should_speak_first = True
                logger.info("Dialin mode: bot will speak first")

        # This handler applies to all modes
        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}")
            # Set next pipeline to None to prevent further pipeline switching
            self.next_pipeline = None

    def get_tools(self, fruit_name):
        """Get the function tools definition for a specific pipeline."""
        if fruit_name == "apple":
            # Apple pipeline has access only to banana and orange functions (and terminate)
            return [
                {
                    "function_declarations": [
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
        else:
            # Banana and orange pipelines only have access to terminate_call
            return [
                {
                    "function_declarations": [
                        {
                            "name": "terminate_call",
                            "description": "Call this function to terminate the call.",
                        },
                    ]
                }
            ]

    def get_system_instruction(self, fruit_name):
        """Get system instruction for a specific fruit pipeline."""
        if fruit_name == "apple":
            return """
Always respond with the word 'Apple'.

You have access to the following functions that you can call:
- respond_with_banana: Call this function when the user asks about bananas.
- respond_with_orange: Call this function when the user asks about oranges.
- terminate_call: Call this function to terminate the call.

If the user mentions bananas or oranges, call the appropriate function instead of responding with 'Apple'.
If the user asks to terminate the call, call the terminate_call function.
"""
        elif fruit_name == "banana":
            return """
Always respond with the word 'Banana'.

You have access to the following function that you can call:
- terminate_call: Call this function to terminate the call.

If the user asks to terminate the call, call the terminate_call function.
"""
        elif fruit_name == "orange":
            return """
Always respond with the word 'Orange'.

You have access to the following function that you can call:
- terminate_call: Call this function to terminate the call.

If the user asks to terminate the call, call the terminate_call function.
"""

    def get_model_for_fruit(self, fruit_name):
        """Get the appropriate model for a fruit pipeline."""
        if fruit_name == "apple":
            return "models/gemini-2.0-flash-lite"
        else:
            return "models/gemini-2.0-flash"  # Using standard flash model for banana and orange

    async def switch_to_banana(
        self, function_name, tool_call_id, args, llm, context, result_callback
    ):
        """Function to switch to banana pipeline."""
        logger.info("Function called: respond_with_banana")

        # If we're already on banana, no need to switch
        if self.current_pipeline == "banana":
            return

        # Set next pipeline
        self.next_pipeline = "banana"

        # Stop current LLM
        await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)

    async def switch_to_orange(
        self, function_name, tool_call_id, args, llm, context, result_callback
    ):
        """Function to switch to orange pipeline."""
        logger.info("Function called: respond_with_orange")

        # If we're already on orange, no need to switch
        if self.current_pipeline == "orange":
            return

        # Set next pipeline
        self.next_pipeline = "orange"

        # Stop current LLM
        await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)

    async def terminate_call(
        self, function_name, tool_call_id, args, llm, context, result_callback
    ):
        """Function to terminate the call."""
        logger.info("Function called: terminate_call")
        await llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def create_pipeline(self, fruit_name):
        """Create a new pipeline for a specific fruit."""
        logger.info(f"Creating {fruit_name} pipeline")

        # Get system instruction and model
        system_instruction = self.get_system_instruction(fruit_name)
        model = self.get_model_for_fruit(fruit_name)

        # Create LLM with tools specific to this pipeline
        llm = GoogleLLMService(
            model=model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            system_instruction=system_instruction,
            tools=self.get_tools(fruit_name),
        )

        # Create context and aggregator
        context = GoogleLLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Register only relevant functions for this pipeline
        if fruit_name == "apple":
            # Apple pipeline has access to banana and orange functions
            for func_name, func in self.fruit_functions.items():
                llm.register_function(func_name, func)
            logger.info(f"Registered banana and orange functions for {fruit_name} pipeline")

        # All pipelines have access to terminate_call
        llm.register_function("terminate_call", self.terminate_call)
        logger.info(f"Registered terminate_call function for {fruit_name} pipeline")

        # Create pipeline components
        components = [self.transport.input()]

        # Special case for apple: always use audio collector regardless of self.use_transcription
        if fruit_name == "apple":
            # Create audio collector for apple pipeline
            audio_collector = UserAudioCollector(context, context_aggregator.user())
            components.append(audio_collector)
            logger.info(f"Using audio collector for {fruit_name} pipeline")
        else:
            # For all other fruits, always use STT
            components.append(self.stt)
            logger.info(f"Using STT for {fruit_name} pipeline")

        # Add remaining components
        components.extend(
            [
                context_aggregator.user(),
                llm,
                self.tts,
                self.transport.output(),
                context_aggregator.assistant(),
            ]
        )

        # Create pipeline and task
        pipeline = Pipeline(components)
        pipeline_task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
        )

        # Set initial context
        context_aggregator.user().set_messages(
            [
                {
                    "role": "system",
                    "content": system_instruction,
                }
            ]
        )

        return pipeline_task, context_aggregator

    async def run_pipeline(self, fruit_name):
        """Run a pipeline for a specific fruit."""
        # Update current pipeline
        self.current_pipeline = fruit_name
        self.next_pipeline = None

        # Log pipeline start
        logger.info(f"=======================================")
        logger.info(f"Starting {fruit_name.upper()} pipeline")
        logger.info(f"=======================================")

        # Create pipeline
        pipeline_task, context_aggregator = await self.create_pipeline(fruit_name)

        # Queue initial context frame in specific cases:
        # 1. For non-apple pipelines (banana/orange) - always queue to trigger immediate response
        # 2. For apple pipeline in dialin mode - queue to make bot speak first
        if fruit_name != "apple":
            logger.info(f"Queueing initial context frame for {fruit_name} pipeline")
            await pipeline_task.queue_frames([context_aggregator.user().get_context_frame()])
        else:
            # For apple pipeline, check if we should speak first (dialin mode)
            if not self.dialout_number and not self.detect_voicemail:
                logger.info(
                    f"Dialin mode: Queueing initial context frame for {fruit_name} pipeline to make bot speak first"
                )
                # In dialin mode, bot speaks first
                await pipeline_task.queue_frames([context_aggregator.user().get_context_frame()])
            else:
                logger.info(
                    f"Apple pipeline - not queueing initial context frame (waiting for user input)"
                )

        # Run pipeline task - this will block until pipeline ends
        await self.runner.run(pipeline_task)

        # Log pipeline end
        logger.info(f"=======================================")
        logger.info(f"{fruit_name.upper()} pipeline completed")
        logger.info(f"=======================================")

        # If a next pipeline is set, run it
        if self.next_pipeline:
            next_pipeline = self.next_pipeline
            self.next_pipeline = None
            await self.run_pipeline(next_pipeline)


async def main(
    room_url: str,
    token: str,
    callId: str,
    callDomain: str,
    detect_voicemail: bool,
    dialout_number: Optional[str],
):
    try:
        # Log startup
        logger.info("=======================================")
        logger.info("Starting dynamic pipeline chatbot")
        logger.info("=======================================")

        if callId != "None" and callDomain != "None":
            dialin_settings = DailyDialinSettings(call_id=callId, call_domain=callDomain)
            # Set up transport
            logger.info("Setting up transport")
            transport_params = DailyParams(
                api_url=daily_api_url,
                api_key=daily_api_key,
                dialin_settings=dialin_settings,
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            )
        else:
            transport_params = DailyParams(
                api_url=daily_api_url,
                api_key=daily_api_key,
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            )

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            transport_params,
        )

        # Set up TTS and STT services
        logger.info("Setting up TTS and STT services")
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Create pipeline runner
        logger.info("Creating pipeline runner")
        runner = PipelineRunner()

        # Create pipeline manager
        logger.info("Creating pipeline manager")
        pipeline_manager = DynamicPipelineManager()
        pipeline_manager.initialize(
            transport=transport,
            tts=tts,
            stt=stt,
            runner=runner,
            dialout_number=dialout_number,
            detect_voicemail=detect_voicemail,
        )

        # Start with apple pipeline
        logger.info("Starting with apple pipeline")
        await pipeline_manager.run_pipeline("apple")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Dynamic Pipeline Chatbot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-d", type=str, help="Call Domain")
    parser.add_argument("-v", action="store_true", help="Detect voicemail")
    parser.add_argument("-o", type=str, help="Dialout number", default=None)
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.d, config.v, config.o))
