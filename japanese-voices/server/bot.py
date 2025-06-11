#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.frames.frames import (
    EndTaskFrame,
    EndFrame,
    StopTaskFrame,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
)
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams


from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)
from pipecatcloud.agent import DailySessionArguments


load_dotenv(override=True)

IS_LOCAL_RUN = os.environ.get("LOCAL_RUN", "0") == "1"

# Configure logger - safely remove default handler if it exists
try:
    logger.remove()
except ValueError:
    pass  # Handler 0 doesn't exist, which is fine
logger.add(sys.stderr, level="DEBUG")


class DynamicPipelineManager:
    """Manages multiple pipelines that can be switched between."""

    def __init__(self):
        self.current_pipeline = None
        self.next_pipeline = None  # Add separate variable for next pipeline
        self.transport = None
        self.tts = None
        self.stt = None
        self.runner = None
        # Add these to store current pipeline references
        self.current_pipeline_task = None
        self.current_context_aggregator = None
        self.current_rtvi = None
        self.current_provider = "openai"
        self.current_voice = "male"

    def initialize(self, transport, tts, stt, runner):
        """Initialize the pipeline manager with transport and services."""
        self.transport = transport
        self.tts = tts
        self.stt = stt
        self.runner = runner

        # Set up appropriate handlers based on the call mode
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handlers based on dial mode."""

        @self.transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"First participant joined: {participant}")

        # This handler applies to all modes
        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}")
            # Set next pipeline to None to prevent further pipeline switching
            self.next_pipeline = None

            # Queue EndFrame to current pipeline task if it exists
            if self.current_pipeline_task:
                logger.info("Queueing EndFrame due to participant leaving")
                await self.current_pipeline_task.queue_frames([EndFrame()])

    def _setup_rtvi_handlers(self, rtvi, pipeline_task, context_aggregator):
        """Set up RTVI event handlers for the current pipeline."""

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("RTVI client is ready - setting bot ready")
            await rtvi.set_bot_ready()

    def get_LLM_for_pipeline(self, pipeline_name, system_instruction, voice):
        """Get the LLM service for a specific pipeline pipeline."""
        if pipeline_name == "google":
            # google pipeline uses Google LLM
            tools = ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="switch_pipeline",
                        description="Switch to a different pipeline.",
                        properties={
                            "pipeline_name": {
                                "type": "string",
                                "enum": ["google", "other"],
                                "description": "The name of the pipeline to switch to.",
                            },
                            "provider": {
                                "type": "string",
                                "description": "The TTS provider to switch to. Available options are: Cartesia, ElevenLabs, Azure, OpenAI.",
                                "enum": ["cartesia", "elevenlabs", "azure", "openai"],
                            },
                            "voice": {
                                "type": "string",
                                "description": "The voice to switch to. Available options are male or female",
                                "enum": ["male", "female"],
                            },
                        },
                        required=["pipeline_name", "provider", "voice"],
                    ),
                    FunctionSchema(
                        name="terminate_call",
                        description="Terminate the call.",
                        properties={},
                        required=[],
                    ),
                ]
            )
            gemini_llm = GeminiMultimodalLiveLLMService(
                api_key=os.getenv("GOOGLE_API_KEY"),
                voice_id=voice,
                system_instruction=system_instruction,
                tools=tools,
                model="models/gemini-2.5-flash-preview-native-audio-dialog",
                # transcribe_user_audio=True,
            )
            return gemini_llm
        else:
            # cartesia and openai pipelines use OpenAI LLM
            return OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    async def switch_pipeline(self, params: FunctionCallParams):
        """Function to switch to a different pipeline pipeline based on user input."""
        pipeline_name = params.arguments["pipeline_name"]
        self.next_pipeline = pipeline_name  # Use separate variable for target
        if self.current_pipeline == "google":
            self.current_provider = params.arguments["provider"]
            self.current_voice = params.arguments["voice"]
        logger.info(
            f"Function called: switch_pipeline with params: {params}, pipeline_name: {pipeline_name}"
        )

        # Stop current LLM
        await params.llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)

    async def switch_provider(self, params: FunctionCallParams):
        """Switch the TTS provider."""
        self.current_provider = params.arguments["provider"]

    async def switch_voice(self, params: FunctionCallParams):
        """Switch to the first voice."""
        self.current_voice = params.arguments["voice"]

    async def terminate_call(self, params: FunctionCallParams):
        """Function to terminate the call."""
        logger.info("Function called: terminate_call")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def create_pipeline(self, pipeline_name):
        """Create a new pipeline for a specific pipeline."""
        logger.info(f"Creating {pipeline_name} pipeline")

        gemini_system_instruction = """
            - You are a helpful assistant that can switch TTS pipelines.
            - The user will only speak English to you. You will always respond in English.
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
            - If the user asks to switch to Cartesia, OpenAI, Azure or Elevenlabs, call the switch_pipeline function with "other", as the pipeline name, the provider name, and the voice.
            - If the user doesn't specify a provider, ask them which provider they want to switch to.
            - If the user doesn't specify a voice, ask them which voice they want to switch to.
            - If the user asks to switch to a female or male voice without asking to switch provider, tell them that Google does not support voice switching.
            
            Rules:
            1. Always respond in English, unless reading the story.
            2. If the user asks to switch to Cartesia, OpenAI, Azure or Elevenlabs, call the switch_pipeline function with "other", as the pipeline name, the provider name, and the voice.
            3. If the user doesn't specify a voice, ask them which voice they want to switch to.
            4. If the user asks to switch to a female or male voice without asking to switch provider, tell them that Google does not support voice switching.
            5. openai, cartesia, azure and elevenlabs are not valid pipeline names. Only google or other are valid pipeline names.
            """

        openai_system_instruction = """
            - You are a helpful assistant that can switch TTS pipelines.
            - The user will only speak English to you. You will always respond in English.
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
            - If the user asks to switch to Cartesia, OpenAI, Azure or Elevenlabs, call the switch_provider function with the provider name.
            - If the user asks to switch to Google, call the switch_pipeline function with "google", as the pipeline name, also specify google as the provider name, and male as the voice.
            - If the user asks to switch to a female or male voice, call the switch_voice function with either male or female as the voice.

            Rules:
            1. Always respond in English, unless reading the story.
            2. If the user asks to switch to Cartesia, OpenAI, Azure or Elevenlabs, call the switch_provider function with the provider name.
            3. If the user asks to switch to Google, call the switch_pipeline function with "google", as the pipeline name, also specify google as the provider name, and male as the voice.
            4. If the user asks to switch to a female or male voice, call the switch_voice function with either male or female as the voice.
            """

        voice = os.getenv("GEMINI_VOICE", "Puck")  # Aoede, Charon, Fenrir, Kore, Puck

        # Create LLM
        llm = self.get_LLM_for_pipeline(pipeline_name, gemini_system_instruction, voice)

        # Create context and aggregator
        if pipeline_name == "google":
            # For google (Gemini), use minimal context - let system instruction handle behavior
            context = OpenAILLMContext()
        else:
            # For non-google (OpenAI), set up tools properly
            tools = ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="switch_pipeline",
                        description="Switch to a different pipeline.",
                        properties={
                            "pipeline_name": {
                                "type": "string",
                                "enum": ["google", "other"],
                                "description": "The name of the pipeline to switch to.",
                            }
                        },
                        required=[
                            "pipeline_name",
                        ],
                    ),
                    FunctionSchema(
                        name="terminate_call",
                        description="Terminate the call.",
                        properties={},
                        required=[],
                    ),
                    FunctionSchema(
                        name="switch_provider",
                        description="Call this function to switch the TTS provider. Either Cartesia, ElevenLabs, Azure or OpenAI.",
                        properties={
                            "provider": {
                                "type": "string",
                                "description": "The TTS provider to switch to. Available options are: Cartesia, ElevenLabs, Azure, OpenAI.",
                                "enum": ["cartesia", "elevenlabs", "azure", "openai"],
                            }
                        },
                        required=["provider"],
                    ),
                    FunctionSchema(
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
                    ),
                ]
            )

            context = OpenAILLMContext(
                [
                    {
                        "role": "system",
                        "content": openai_system_instruction,
                    }
                ],
                tools,  # Only for OpenAI
            )

        context_aggregator = llm.create_context_aggregator(context)

        # Register functions with the LLM
        if pipeline_name == "google":
            # For Google pipeline, register functions with the LLM
            llm.register_function("switch_pipeline", self.switch_pipeline)
            llm.register_function("terminate_call", self.terminate_call)
        else:
            llm.register_function("switch_pipeline", self.switch_pipeline)
            llm.register_function("switch_provider", self.switch_provider)
            llm.register_function("switch_voice", self.switch_voice)
            llm.register_function("terminate_call", self.terminate_call)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Create pipeline components
        components = [self.transport.input()]

        cartesia_model = os.getenv("CARTESIA_MODEL", "sonic-2")
        cartesia_male_voice = os.getenv(
            "CARTESIA_MALE_VOICE", "06950fa3-534d-46b3-93bb-f852770ea0b5"
        )
        cartesia_male = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY", ""),
            model=cartesia_model,
            voice_id=cartesia_male_voice,  # male voice
        )

        cartesia_female_voice = os.getenv(
            "CARTESIA_FEMALE_VOICE", "59d4fd2f-f5eb-4410-8105-58db7661144f"
        )
        cartesia_female = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY", ""),
            model=cartesia_model,
            voice_id=cartesia_female_voice,  # female voice
        )

        elevenlabs_male_voice = os.getenv(
            "ELEVENLABS_MALE_VOICE", "GxxMAMfQkDlnqjpzjLHH"
        )
        elevenlabs_male = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=elevenlabs_male_voice,  # male voice
        )  # https://play.cartesia.ai/voices?language=ja

        elevenlabs_female_voice = os.getenv(
            "ELEVENLABS_FEMALE_VOICE", "RBnMinrYKeccY3vaUxlZ"
        )
        elevenlabs_female = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=elevenlabs_female_voice,  # female voice
        )  # https://elevenlabs.io/app/voice-library?search=GxxMAMfQkDlnqjpzjLHH

        azure_male_language = os.getenv("AZURE_MALE_LANGUAGE", "ja-JP")
        azure_male_rate = os.getenv("AZURE_MALE_RATE", "1.1")
        azure_male_style = os.getenv("AZURE_MALE_STYLE", "default")

        azure_male = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            # voice="ja-JP-ja-JP-KeitaNeural",  # male voice
            voice=os.getenv("AZURE_MALE_VOICE", "ja-JP-ja-JP-KeitaNeural"),
            params=AzureTTSService.InputParams(
                language=azure_male_language,
                rate=azure_male_rate,
                style=azure_male_style,
            ),
        )  # https://speech.microsoft.com/portal/voicegallery

        azure_female_language = os.getenv("AZURE_FEMALE_LANGUAGE", "ja-JP")
        azure_female_rate = os.getenv("AZURE_FEMALE_RATE", "1.1")
        azure_female_style = os.getenv("AZURE_FEMALE_STYLE", "default")
        azure_female = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="ja-JP-NanamiNeural",  # female voice
            params=AzureTTSService.InputParams(
                language=azure_female_language,
                rate=azure_female_rate,
                style=azure_female_style,
            ),
        )

        openai_male_voice = os.getenv("OPENAI_MALE_VOICE", "alloy")
        openai_male = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=openai_male_voice,  # male voice
        )  # https://www.openai.fm/ / https://platform.openai.com/docs/guides/text-to-speech#supported-languages

        openai_female_voice = os.getenv("OPENAI_FEMALE_VOICE", "nova")
        openai_female = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=openai_female_voice,  # female voice
        )

        async def cartesia_filter_male(frame) -> bool:
            return self.current_provider == "cartesia" and self.current_voice == "male"

        async def cartesia_filter_female(frame) -> bool:
            return (
                self.current_provider == "cartesia" and self.current_voice == "female"
            )

        async def elevenlabs_filter_male(frame) -> bool:
            return (
                self.current_provider == "elevenlabs" and self.current_voice == "male"
            )

        async def elevenlabs_filter_female(frame) -> bool:
            return (
                self.current_provider == "elevenlabs" and self.current_voice == "female"
            )

        async def azure_filter_male(frame) -> bool:
            return self.current_provider == "azure" and self.current_voice == "male"

        async def azure_filter_female(frame) -> bool:
            return self.current_provider == "azure" and self.current_voice == "female"

        async def openai_filter_male(frame) -> bool:
            return self.current_provider == "openai" and self.current_voice == "male"

        async def openai_filter_female(frame) -> bool:
            return self.current_provider == "openai" and self.current_voice == "female"

        parallel_pipeline_components = ParallelPipeline(
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
        )

        # Special case for google: uses native audio
        if pipeline_name == "google":
            logger.debug("Google pipeline - using native audio")
            components.extend(
                [
                    rtvi,
                    context_aggregator.user(),
                    llm,
                    self.transport.output(),
                    context_aggregator.assistant(),
                ]
            )
            logger.info("Setting up native audio pipeline")
        else:
            # For all other pipelines, use STT + TTS
            logger.debug("Non-google pipeline - using STT + TTS")
            components.extend(
                [
                    rtvi,
                    self.stt,
                    context_aggregator.user(),
                    llm,
                    parallel_pipeline_components,
                    self.transport.output(),
                    context_aggregator.assistant(),
                ]
            )
            logger.info(f"Using STT for {pipeline_name} pipeline")

        # Create pipeline and task
        pipeline = Pipeline(components)
        pipeline_task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
            observers=[TranscriptionLogObserver(), RTVIObserver(rtvi)],
        )

        # Store references for event handlers
        self.current_pipeline_task = pipeline_task
        self.current_context_aggregator = context_aggregator
        self.current_rtvi = rtvi

        # Set up RTVI event handlers with proper access to pipeline_task and context_aggregator
        self._setup_rtvi_handlers(rtvi, pipeline_task, context_aggregator)

        return pipeline_task, context_aggregator

    async def run_pipeline(self, pipeline_name):
        """Run a pipeline for a specific pipeline."""
        # Check if this is the first pipeline or a switch
        is_first_pipeline = self.current_pipeline is None
        logger.info(f"Is first pipeline: {is_first_pipeline}")

        # Update current pipeline
        self.current_pipeline = pipeline_name

        # Log pipeline start
        logger.info(f"=======================================")
        logger.info(f"Starting {pipeline_name.upper()} pipeline")
        logger.info(f"=======================================")

        # Create pipeline
        pipeline_task, context_aggregator = await self.create_pipeline(pipeline_name)

        # Run pipeline task - this will block until pipeline ends
        await self.runner.run(pipeline_task)

        # Log pipeline end
        logger.info(f"=======================================")
        logger.info(f"{pipeline_name.upper()} pipeline completed")
        logger.info(f"=======================================")

        # If a next pipeline is set, run it
        if self.next_pipeline:
            next_pipeline = self.next_pipeline
            self.next_pipeline = None  # Reset the next pipeline
            logger.info(f"Switching to next pipeline: {next_pipeline}")
            await self.run_pipeline(next_pipeline)


async def run_bot(
    room_url: str,
    token: str,
    config: dict,
) -> None:
    try:
        # Log startup
        logger.info("=======================================")
        logger.info("Starting dynamic pipeline chatbot")
        logger.info("=======================================")

        if not IS_LOCAL_RUN:
            from pipecat.audio.filters.krisp_filter import KrispFilter

        transport_params = DailyParams(
            audio_in_enabled=True,
            audio_in_filter=None if IS_LOCAL_RUN else KrispFilter(),
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
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
        )

        # Start with other pipeline (non-google)
        logger.info("Starting with other pipeline")
        await pipeline_manager.run_pipeline("other")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback

        logger.error(traceback.format_exc())


async def bot(args: DailySessionArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    logger.info(f"Bot process initialized {args.room_url} {args.token}")

    try:
        await run_bot(args.room_url, args.token, args.body)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise
