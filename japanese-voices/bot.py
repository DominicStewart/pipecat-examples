#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

## japanese-voices/bot.py ##

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from deepgram import LiveOptions

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
    DailyTranscriptionSettings,
)
from pipecat.services.google.google import GoogleLLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
)


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class UserAudioCollector(FrameProcessor):
    """Collects audio frames in a buffer, then adds them to the LLM context when the user stops speaking."""

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
            # Skip transcription frames - we're handling audio directly
            return
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                # When speaking, collect frames
                self._audio_frames.append(frame)
            else:
                # Maintain a rolling buffer of recent audio (for start of speech)
                self._audio_frames.append(frame)
                frame_duration = (
                    len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                )
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


async def run_bot(
    room_url: str,
    token: str,
    daily_api_key: str,
    daily_api_url: str,
) -> None:
    """Run the bot with the provided parameters."""
    # Initialize the transport

    params = DailyParams(
        api_key=daily_api_key,
        api_url=daily_api_url,
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
        transcription_enabled=True,
        transcription_settings=DailyTranscriptionSettings(
            language="multi",
            extra={
                "mip_opt_out": True,
                "keywords": ["Mustang:5", "Kwindla:5", "Snuffleupagus:10"],
            },
        ),
    )

    transport = DailyTransport(room_url, token, "Japanese Bot", params)

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(language="multi"),
    )

    # Initialize the TTS service
    cartesia_male = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="06950fa3-534d-46b3-93bb-f852770ea0b5",  # male voice
    )

    cartesia_female = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="59d4fd2f-f5eb-4410-8105-58db7661144f",  # female voice
    )

    elevenlabs_male = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id="GxxMAMfQkDlnqjpzjLHH",  # male voice
    )  # https://play.cartesia.ai/voices?language=ja

    elevenlabs_female = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id="RBnMinrYKeccY3vaUxlZ",  # female voice
    )  # https://elevenlabs.io/app/voice-library?search=GxxMAMfQkDlnqjpzjLHH

    azure_male = AzureTTSService(
        api_key=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        voice="ja-JP-ja-JP-KeitaNeural",  # male voice
        params=AzureTTSService.InputParams(
            language="ja-JP", rate="1.1", style="default"
        ),
    )  # https://speech.microsoft.com/portal/voicegallery

    azure_female = AzureTTSService(
        api_key=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        voice="ja-JP-NanamiNeural",  # female voice
        params=AzureTTSService.InputParams(
            language="ja-JP", rate="1.1", style="default"
        ),
    )

    openai_male = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="alloy",  # male voice
    )  # https://www.openai.fm/ / https://platform.openai.com/docs/guides/text-to-speech#supported-languages

    openai_female = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="nova",  # female voice
    )

    # Initialize the LLM service
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    current_provider = "elevenlabs"
    current_voice = "male"

    async def switch_voice(params: FunctionCallParams):
        """Switch to the first voice."""
        nonlocal current_voice
        current_voice = params.arguments["voice"]
        await params.result_callback(
            {
                # 英語 → 日本語に変更
                # "voice": f"Switched to {current_voice} voice",
                "voice": f"{current_voice} 音声に切り替えました",
            }
        )

    async def switch_tts(params: FunctionCallParams):
        """Switch the TTS provider."""
        nonlocal current_provider
        current_provider = params.arguments["provider"]
        await params.result_callback(
            {
                # 英語 → 日本語に変更
                # "provider": f"Switched to {current_provider} TTS provider",
                "provider": f"{current_provider} のTTSプロバイダーに切り替えました",
            }
        )

    async def switch_pipeline(params: FunctionCallParams):
        """Switch the pipeline based on the current provider and voice."""
        await params.llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)

    switch_voice_function = FunctionSchema(
        name="switch_voice",
        description="Call this function to switch the voice used by the TTS provider.",
        properties={
            "voice": {
                "type": "string",
                "description": "The voice to switch to. Available options are male or female",
                "enum": ["male", "female"],
            }
        },
        required=["voice"],
    )

    switch_pipeline_function = FunctionSchema(
        name="switch_pipeline",
        description="Call this function to switch the pipeline.",
        properties={},
        required=[],
    )

    switch_tts_function = FunctionSchema(
        name="switch_tts",
        description="Call this function to switch the TTS provider.",
        properties={
            "provider": {
                "type": "string",
                "description": "The TTS provider to switch to. Available options are: Cartesia, ElevenLabs, Azure, OpenAI.",
                # Cartesia, ElevenLabs, Azure, OpenAI のいずれか
                "enum": ["cartesia", "elevenlabs", "azure", "openai"],
            }
        },
        required=["provider"],
    )

    llm.register_function("switch_voice", switch_voice)
    llm.register_function("switch_tts", switch_tts)
    llm.register_function("switch_pipeline", switch_pipeline)

    tools = ToolsSchema(
        standard_tools=[
            switch_voice_function,
            switch_tts_function,
            switch_pipeline_function,
        ]
    )

    messages_jp = [
        {
            "role": "system",
            "content": """
            - You are a helpful assistant that can switch TTS providers, voices or pipelines.
            - The user will only speak English to you. But you will always respond in Japanese.
            - Please ask the user if they want to hear a story, and if they say yes, read the excerpt from "吾輩は猫である" in Japanese.
            - If the user asks you to tell them a story, you will read the following excerpt from "吾輩は猫である" in Japanese.
            "吾輩は猫である":
                吾輩は猫である。名前はまだ無い。
                どこで生れたかとんと見当がつかぬ。
                何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
                吾輩はここで始めて人間というものを見た。
                しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
                この書生というのは時々我々を捕えて煮て食うという話である。
                しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
                ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
            - If the user says "No", continue the conversation normally.
            - Each TTS provider has a male or female voice available. You can switch between them.
            - If the user asks to switch voices, you will call the switch_voice function with the appropriate voice.
            - If the user does not specify male or female, please ask them which voice they prefer.
            - There are four TTS providers available: Cartesia, ElevenLabs, Azure, and OpenAI.
            - If the user asks to switch TTS providers, you will call the switch_tts function with the appropriate provider.
            - If the user does not specify a provider, please ask them which provider they prefer.
            - If the user asks to switch pipelines, you will call the switch_pipeline function.
            
            Rules:
            1. Always respond in Japanese.
            2. If the user asks to hear a story, read the excerpt from "吾輩は猫である" in Japanese.
            3. If the user asks to switch voices, call the switch_voice function with the appropriate voice.
            4. If the user asks to switch TTS providers, call the switch_tts function with the appropriate provider.
            5. If the user asks to switch pipelines, call the switch_pipeline function.
            6. If the user asks to switch voices or TTS providers, always ask them which voice or provider they prefer.
            """,
        }
    ]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages_jp, tools)
    context_aggregator = llm.create_context_aggregator(context)

    async def cartesia_filter_male(frame) -> bool:
        return current_provider == "cartesia" and current_voice == "male"

    async def cartesia_filter_female(frame) -> bool:
        return current_provider == "cartesia" and current_voice == "female"

    async def elevenlabs_filter_male(frame) -> bool:
        return current_provider == "elevenlabs" and current_voice == "male"

    async def elevenlabs_filter_female(frame) -> bool:
        return current_provider == "elevenlabs" and current_voice == "female"

    async def azure_filter_male(frame) -> bool:
        return current_provider == "azure" and current_voice == "male"

    async def azure_filter_female(frame) -> bool:
        return current_provider == "azure" and current_voice == "female"

    async def openai_filter_male(frame) -> bool:
        return current_provider == "openai" and current_voice == "male"

    async def openai_filter_female(frame) -> bool:
        return current_provider == "openai" and current_voice == "female"

    # Create the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            ParallelPipeline(
                [
                    FunctionFilter(cartesia_filter_male),
                    cartesia_male,
                ],  # Cartesia Male Japanese Voice
                [
                    FunctionFilter(cartesia_filter_female),
                    cartesia_female,
                ],  # Cartesia Female Japanese Voice
                [
                    FunctionFilter(elevenlabs_filter_male),
                    elevenlabs_male,
                ],  # ElevenLabs Male Japanese Voice
                [
                    FunctionFilter(elevenlabs_filter_female),
                    elevenlabs_female,
                ],  # ElevenLabs Female Japanese Voice
                [
                    FunctionFilter(azure_filter_male),
                    azure_male,
                ],  # Azure Male Japanese Voice
                [
                    FunctionFilter(azure_filter_female),
                    azure_female,
                ],  # Azure Female Japanese Voice
                [
                    FunctionFilter(openai_filter_male),
                    openai_male,
                ],  # OpenAI Male Japanese Voice
                [
                    FunctionFilter(openai_filter_female),
                    openai_female,
                ],  # OpenAI Female Japanese Voice
            ),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        observers=[TranscriptionLogObserver()],
    )

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        # await transport.capture_participant_transcription(participant["id"])
        messages_jp.append(
            {
                "role": "system",
                "content": (
                    """
                    - You are a helpful assistant that tells stories in Japanese.
                    - The user will only speak English to you. But you will always respond in Japanese.
                    - Please ask the user if they want to hear a story, and if they say yes, read the excerpt from "吾輩は猫である" in Japanese.
                    - If the user asks you to tell them a story, you will read the following excerpt from "吾輩は猫である" in Japanese.
                    "吾輩は猫である":
                        吾輩は猫である。名前はまだ無い。
                        どこで生れたかとんと見当がつかぬ。
                        何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
                        吾輩はここで始めて人間というものを見た。
                        しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
                        この書生というのは時々我々を捕えて煮て食うという話である。
                        しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
                        ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
                    - If the user says "No", continue the conversation normally.                    
                    Rules:
                    1. Always respond in Japanese.
                    2. If the user asks to hear a story, read the excerpt from "吾輩は猫である" in Japanese."""
                ),
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    # Create the pipeline runner
    runner = PipelineRunner()

    # Start the pipeline runner
    await runner.run(task)

    # ---------------- NEXT PIPELINE ----------------

    gemini_pipeline_system_instructions = """
    - You are a helpful assistant that tells stories in Japanese.
    - The user will only speak English to you. But you will always respond in Japanese.
    - Start by telling them the Pipeline has now changed in Japanese.
    - Now read this excerpt from "吾輩は猫である" in Japanese:
    "吾輩は猫である":
    吾輩は猫である。名前はまだ無い。
    どこで生れたかとんと見当がつかぬ。
    何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
    吾輩はここで始めて人間というものを見た。
    しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
    この書生というのは時々我々を捕えて煮て食うという話である。
    しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
    ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
    - If the user says "No", continue the conversation normally.                    
    Rules:
    1. Always respond in Japanese.
    2. If the user asks to hear a story, read the excerpt from "吾輩は猫である" in Japanese."""

    gemini_llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        # voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        system_instruction=gemini_pipeline_system_instructions,
        # model="gemini-2.5-flash-preview-05-20",
        model="models/gemini-2.5-flash-preview-native-audio-dialog",
        transcribe_user_audio=True,
    )

    # gemini_context = GoogleLLMContext()
    gemini_context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": """
                - You are a helpful assistant that tells stories in Japanese.
                - The user will only speak English to you. But you will always respond in Japanese.
                - Start by telling them the Pipeline has now changed in Japanese.
                - Now read this excerpt from "吾輩は猫である" in Japanese:
                "吾輩は猫である":
                吾輩は猫である。名前はまだ無い。
                どこで生れたかとんと見当がつかぬ。
                何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
                吾輩はここで始めて人間というものを見た。
                しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
                この書生というのは時々我々を捕えて煮て食うという話である。
                しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
                ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
            - If the user says "No", continue the conversation normally.                    
            Rules:
            1. Always respond in Japanese.
            2. If the user asks to hear a story, read the excerpt from "吾輩は猫である" in Japanese.""",
            }
        ]
    )
    gemini_context_aggregator = gemini_llm.create_context_aggregator(gemini_context)

    gemini_audio_collector = UserAudioCollector(
        gemini_context, gemini_context_aggregator.user()
    )

    gemini_pipeline = Pipeline(
        [
            transport.input(),
            # gemini_audio_collector,
            gemini_context_aggregator.user(),
            gemini_llm,
            transport.output(),
            gemini_context_aggregator.assistant(),
        ]
    )

    gemini_pipeline_task = PipelineTask(
        gemini_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Update participant left handler for human conversation phase
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())
        await gemini_pipeline_task.queue_frame(EndFrame())

    # gemini_context_aggregator.user().set_messages(
    #     [
    #         {
    #             "role": "system",
    #             "content": """
    #             - You are a helpful assistant that tells stories in Japanese.
    #             - The user will only speak English to you. But you will always respond in Japanese.
    #             - Start by telling them the Pipeline has now changed in Japanese.
    #             - Now read this excerpt from "吾輩は猫である" in Japanese:
    #             "吾輩は猫である":
    #             吾輩は猫である。名前はまだ無い。
    #             どこで生れたかとんと見当がつかぬ。
    #             何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
    #             吾輩はここで始めて人間というものを見た。
    #             しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
    #             この書生というのは時々我々を捕えて煮て食うという話である。
    #             しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
    #             ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
    #         - If the user says "No", continue the conversation normally.
    #         Rules:
    #         1. Always respond in Japanese.
    #         2. If the user asks to hear a story, read the excerpt from "吾輩は猫である" in Japanese.""",
    #         }
    #     ]
    # )

    await gemini_pipeline_task.queue_frames(
        [gemini_context_aggregator.user().get_context_frame()]
    )

    await runner.run(gemini_pipeline_task)


async def main():
    """Main function to run the bot."""

    daily_api_key = os.getenv("DAILY_API_KEY", "")
    daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
    room_url = os.getenv("DAILY_ROOM_URL", "")
    token = os.getenv("DAILY_ROOM_TOKEN", "")

    if not daily_api_key:
        logger.error("DAILY_API_KEY environment variable is not set.")
        return

    if not daily_api_url:
        logger.error("DAILY_API_URL environment variable is not set.")
        return

    if not room_url or not token:
        logger.error("Room URL or token is not provided.")
        return

    await run_bot(
        room_url,
        token,
        daily_api_key,
        daily_api_url,
    )


if __name__ == "__main__":
    asyncio.run(main())
