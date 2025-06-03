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
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
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


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
        vad_analyzer=SileroVADAnalyzer(),
        # transcription_enabled=True,
        # transcription_settings=DailyTranscriptionSettings(
        #    language="multi", model="nova-3"
        # ),
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

    switch_voice_function = FunctionSchema(
        name="switch_voice",
        description="TTSプロバイダーの音声を切り替えます",
        properties={
            "voice": {
                "type": "string",
                "description": "切り替える音声を指定します（'male' または 'female'）",
                "enum": ["male", "female"],
            }
        },
        required=["voice"],
    )

    switch_tts_function = FunctionSchema(
        name="switch_tts",
        description="利用するTTSプロバイダーを切り替えます",
        properties={
            "provider": {
                "type": "string",
                "description": "切り替えるTTSプロバイダーを指定します",
                # Cartesia, ElevenLabs, Azure, OpenAI のいずれか
                "enum": ["cartesia", "elevenlabs", "azure", "openai"],
            }
        },
        required=["provider"],
    )

    llm.register_function("switch_voice", switch_voice)
    llm.register_function("switch_tts", switch_tts)

    tools = ToolsSchema(standard_tools=[switch_voice_function, switch_tts_function])

    messages_jp = [
        {
            "role": "system",
            "content": """
            あなたは、TTSプロバイダーと音声を切り替えることができる役立つアシスタントです。
            以下の動作ルールに従ってください：

            1. 利用可能なTTSプロバイダー：Cartesia、ElevenLabs、Azure、OpenAI。
            各プロバイダーには「male」（男性の声）と「female」（女性の声）の2種類の音声があります。
            - 音声を切り替えるには switch_voice 関数を呼び出してください。
            - プロバイダーを切り替えるには switch_tts 関数を呼び出してください。

            2. ユーザーが最初に参加したら、必ず「物語を聞きたいですか？」と日本語で尋ねてください。
            - ユーザーが「はい」と答えた場合は、以下の文章をすべて読み上げてください：
                ―――――――――――――――――――――――――――
                吾輩は猫である。名前はまだ無い。
                どこで生れたかとんと見当がつかぬ。
                何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
                吾輩はここで始めて人間というものを見た。
                しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
                この書生というのは時々我々を捕えて煮て食うという話である。
                しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
                ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
                ―――――――――――――――――――――――――――
            - ユーザーが「いいえ」または他の回答をした場合は、通常の会話を続けてください。

            3. ユーザーが日本語で「女性の声に切り替えて」などと発話したときは、switch_voice を呼び出して音声を切り替えた後、
            自動的に上記の「吾輩は猫である」の抜粋を読み始めてください。

            4. ユーザーが日本語で「Azureに切り替えて」などと発話したときは、switch_tts を呼び出して
            プロバイダーを切り替えた後、同じく抜粋を読み始めてください。

            5. それ以外の会話はすべて日本語で行ってください。

            6. 返答は常に日本語で、丁寧かつ親しみやすい口調でお願いします。
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
                    """ユーザーが参加したら、まず「物語を聞きたいですか？」と尋ねてください。"
                    もしユーザーが「はい」と答えた場合、以下の文章をそのまま日本語で読み上げてください：
                    吾輩は猫である。名前はまだ無い。
                    どこで生れたかとんと見当がつかぬ。
                    何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
                    吾輩はここで始めて人間というものを見た。
                    しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
                    この書生というのは時々我々を捕えて煮て食うという話である。
                    しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
                    ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
                    
                    ユーザーが「いいえ」または他の回答をした場合は、その旨を受けて次の指示に従ってください。"""
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
