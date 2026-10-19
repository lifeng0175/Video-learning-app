"""Streamlit UI for local video and transcript viewing."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import streamlit as st

from src.app_settings import (
    AppSettings,
    AppSettingsFormatError,
    load_app_settings,
    save_app_settings,
)
from src.asr import load_asr_model
from src.playback import (
    MediaProbeError,
    PlaybackAssessment,
    PlaybackError,
    PlaybackToolNotFoundError,
    assess_playback,
    ensure_playback_proxy,
)
from src.transcription_service import (
    ExistingTranscriptInvalidError,
    InvalidTranscriptionVideoError,
    TranscriptionBusyError,
    TranscriptionResult,
    TranscriptionStage,
    transcribe_video_to_transcript,
)
from src.video_loader import InvalidVideoFileError, UnsupportedVideoFormatError
from src.video_transcript import (
    TranscriptFormatError,
    discover_video_files,
    find_transcript_for_video,
    load_video_transcript,
)
from src.video_upload import import_uploaded_video


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("VIDEO_LEARNING_DATA_DIR", PROJECT_ROOT / "data"))
VIDEOS_DIR = DATA_DIR / "videos"
UPLOAD_STAGING_DIR = DATA_DIR / "cache" / "uploads"
PLAYBACK_CACHE_DIR = DATA_DIR / "cache" / "playback"
METADATA_DIR = DATA_DIR / "metadata"
SETTINGS_PATH = DATA_DIR / "config" / "app_settings.json"
AUTO_TRANSCRIBE_SETTING_KEY = "auto_transcribe_after_import"
LOGGER = logging.getLogger(__name__)

UI_TEXT_JA = {
    "page_title": "動画学習ワークスペース",
    "main_title": "動画・文字起こし",
    "phase_caption": "Phase 1B・英語動画と文字起こしのローカルビューア",
    "import_header": "動画をインポート",
    "auto_transcribe": "動画追加後に自動で文字起こしする",
    "auto_transcribe_help": (
        "新しくインポートした動画だけを自動で文字起こしします。"
        "既存動画の選択や画面の再読み込みでは実行されません。"
    ),
    "settings_load_failed": (
        "設定ファイルを読み込めませんでした。安全のため自動文字起こしを OFF にします。"
        "詳細: {error}"
    ),
    "settings_save_failed": "設定を保存できませんでした。詳細: {error}",
    "upload_label": "MP4 または WebM ファイルを選択",
    "upload_limit": (
        "アップロード上限: {size} MB。インポートが完了するまで、"
        "動画はブラウザのメモリに保持されます。"
    ),
    "import_button": "動画をインポート",
    "importing": "動画をインポートしています…",
    "unsupported_video": "MP4 または WebM 形式の動画を選択してください。",
    "invalid_video": (
        "動画を読み込めませんでした。ファイルが空または破損していないか"
        "確認してください。"
    ),
    "invalid_video_value": "動画ファイルの内容を確認できませんでした。",
    "import_failed": "動画をインポートできませんでした。詳細: {error}",
    "duplicate_video": "同じ動画はすでに登録されています。既存の動画を選択しました。",
    "import_success": "動画をインポートしました。",
    "no_videos": "MP4 または WebM の動画をインポートしてください。",
    "video_selector": "動画",
    "segment_count": "字幕: {count} 件",
    "duration": "長さ: {duration}",
    "transcript_json_invalid": "文字起こし JSON の形式が正しくありません。詳細: {error}",
    "transcript_structure_invalid": (
        "文字起こしデータの構造が正しくありません。詳細: {error}"
    ),
    "transcript_video_missing": "動画ファイルが見つかりません。詳細: {error}",
    "transcript_read_failed": "文字起こしデータを読み込めませんでした。詳細: {error}",
    "playback_tools_missing": (
        "再生互換性を確認するには、FFmpeg と ffprobe を PATH から"
        "実行できるようにしてください。"
    ),
    "playback_probe_failed": "動画の再生互換性を確認できませんでした。詳細: {error}",
    "playback_unknown": (
        "この動画のブラウザ再生互換性は確認されていません"
        "（映像: {video_codec}、音声: {audio_codec}）。"
    ),
    "legacy_video": (
        "この動画には正式な video_id がありません。互換再生用コピーを作成するには、"
        "動画をもう一度インポートしてください。"
    ),
    "video_heading": "動画",
    "proxy_required": (
        "この AV1 WebM をブラウザで再生するには、H.264/AAC の"
        "互換再生用コピーが必要です。"
    ),
    "create_proxy_button": "互換再生用コピーを作成",
    "creating_proxy": "互換再生用コピーを作成しています…",
    "playback_create_failed": "互換再生用コピーを作成できませんでした。詳細: {error}",
    "playback_create_success": "互換再生用コピーを作成しました。",
    "using_cached_proxy": "H.264/AAC の互換再生用コピーを使用しています。",
    "transcript_heading": "文字起こし",
    "not_transcribed": "この動画はまだ文字起こしされていません。",
    "start_transcription": "文字起こしを開始",
    "retry_transcription": "文字起こしを再試行",
    "transcription_preparing": "文字起こしを準備しています…",
    "transcription_loading_model": "ASR モデルを読み込んでいます…",
    "transcription_running": "文字起こし中…",
    "transcription_saving": "文字起こし結果を保存しています…",
    "transcription_success": "文字起こしが完了しました。",
    "transcription_status_failed": "文字起こしに失敗しました。",
    "transcription_failed": (
        "文字起こしに失敗しました。GPU/CUDA とモデルの設定を確認してください。"
        "詳細: {error}"
    ),
    "transcription_busy": (
        "別の文字起こしを実行中です。完了してからもう一度お試しください。"
    ),
    "transcription_video_invalid": (
        "元の動画を読み込めないため、文字起こしを開始できません。"
        "動画ファイルを確認してください。"
    ),
    "transcript_existing_invalid": (
        "既存の文字起こし JSON が破損しているか、動画との関連付けが正しくありません。"
    ),
    "transcript_recovery": (
        "既存の JSON をバックアップしてから修正または削除し、"
        "ページを再読み込みして文字起こしを開始してください。"
    ),
}


def format_timestamp(seconds: float) -> str:
    """Format seconds as an hours-aware transcript timestamp."""
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def video_mime_type(video_path: Path) -> str:
    """Return the browser media type for a supported local video."""
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
    }.get(video_path.suffix.lower(), "video/mp4")


st.set_page_config(
    page_title=UI_TEXT_JA["page_title"],
    page_icon="▶️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_cached_asr_model() -> object:
    """Load and reuse the validated ASR model across Streamlit reruns."""
    return load_asr_model()


def run_transcription_with_status(video_path: Path) -> TranscriptionResult:
    """Run the synchronous service while reporting its real execution stage."""
    status_labels = {
        "preparing": UI_TEXT_JA["transcription_preparing"],
        "loading_model": UI_TEXT_JA["transcription_loading_model"],
        "transcribing": UI_TEXT_JA["transcription_running"],
        "saving": UI_TEXT_JA["transcription_saving"],
        "completed": UI_TEXT_JA["transcription_success"],
    }
    status = st.status(
        UI_TEXT_JA["transcription_preparing"],
        expanded=True,
    )

    def update_status(stage: TranscriptionStage) -> None:
        status.update(label=status_labels[stage], state="running")

    st.session_state["transcription_running_video"] = video_path.name
    try:
        result = transcribe_video_to_transcript(
            video_path,
            METADATA_DIR,
            model_loader=get_cached_asr_model,
            progress_callback=update_status,
        )
    except Exception:
        status.update(
            label=UI_TEXT_JA["transcription_status_failed"],
            state="error",
            expanded=True,
        )
        raise
    else:
        status.update(
            label=UI_TEXT_JA["transcription_success"],
            state="complete",
            expanded=False,
        )
        return result
    finally:
        st.session_state.pop("transcription_running_video", None)


def transcription_error_message(error: Exception) -> str:
    """Return a concise Japanese message for a transcription failure."""
    if isinstance(error, ExistingTranscriptInvalidError):
        return " ".join(
            (
                UI_TEXT_JA["transcript_existing_invalid"],
                UI_TEXT_JA["transcript_recovery"],
            )
        )
    if isinstance(error, TranscriptionBusyError):
        return UI_TEXT_JA["transcription_busy"]
    if isinstance(error, InvalidTranscriptionVideoError):
        return UI_TEXT_JA["transcription_video_invalid"]
    return UI_TEXT_JA["transcription_failed"].format(error=error)


def persist_auto_transcription_setting() -> None:
    """Persist the checkbox value outside Streamlit session state."""
    try:
        save_app_settings(
            AppSettings(
                auto_transcribe_after_import=bool(
                    st.session_state[AUTO_TRANSCRIBE_SETTING_KEY]
                )
            ),
            SETTINGS_PATH,
        )
    except OSError as error:
        LOGGER.exception("Could not save application settings")
        st.session_state["settings_save_error"] = UI_TEXT_JA[
            "settings_save_failed"
        ].format(error=error)
    else:
        st.session_state.pop("settings_save_error", None)


st.title(UI_TEXT_JA["main_title"])
st.caption(UI_TEXT_JA["phase_caption"])

st.sidebar.header(UI_TEXT_JA["import_header"])
try:
    app_settings = load_app_settings(SETTINGS_PATH)
    settings_load_error = None
except (AppSettingsFormatError, OSError, UnicodeError) as error:
    LOGGER.exception("Could not load application settings")
    app_settings = AppSettings(auto_transcribe_after_import=False)
    settings_load_error = UI_TEXT_JA["settings_load_failed"].format(error=error)

if AUTO_TRANSCRIBE_SETTING_KEY not in st.session_state:
    st.session_state[AUTO_TRANSCRIBE_SETTING_KEY] = (
        app_settings.auto_transcribe_after_import
    )
auto_transcribe_after_import = st.sidebar.checkbox(
    UI_TEXT_JA["auto_transcribe"],
    key=AUTO_TRANSCRIBE_SETTING_KEY,
    help=UI_TEXT_JA["auto_transcribe_help"],
    on_change=persist_auto_transcription_setting,
)
if settings_load_error is not None:
    st.sidebar.warning(settings_load_error)
settings_save_error = st.session_state.get("settings_save_error")
if settings_save_error is not None:
    st.sidebar.warning(settings_save_error)
uploaded_file = st.sidebar.file_uploader(
    UI_TEXT_JA["upload_label"],
    type=["mp4", "webm"],
    accept_multiple_files=False,
)
max_upload_size_mb = st.get_option("server.maxUploadSize")
st.sidebar.caption(
    UI_TEXT_JA["upload_limit"].format(size=max_upload_size_mb)
)

if st.sidebar.button(
    UI_TEXT_JA["import_button"],
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
):
    try:
        with st.spinner(UI_TEXT_JA["importing"]):
            import_result = import_uploaded_video(
                uploaded_file,
                uploaded_file.name,
                VIDEOS_DIR,
                UPLOAD_STAGING_DIR,
            )
    except UnsupportedVideoFormatError:
        st.sidebar.error(UI_TEXT_JA["unsupported_video"])
    except InvalidVideoFileError:
        st.sidebar.error(UI_TEXT_JA["invalid_video"])
    except ValueError:
        st.sidebar.error(UI_TEXT_JA["invalid_video_value"])
    except OSError as error:
        st.sidebar.error(UI_TEXT_JA["import_failed"].format(error=error))
    else:
        st.session_state["selected_video_name"] = import_result.stored_file_name
        if import_result.is_duplicate:
            st.session_state.pop("pending_auto_transcription_video", None)
            st.session_state["import_notice"] = (
                "duplicate",
                UI_TEXT_JA["duplicate_video"],
            )
        else:
            if auto_transcribe_after_import:
                st.session_state["pending_auto_transcription_video"] = (
                    import_result.stored_file_name
                )
            else:
                st.session_state.pop("pending_auto_transcription_video", None)
            st.session_state["import_notice"] = (
                "success",
                UI_TEXT_JA["import_success"],
            )
        st.rerun()

import_notice = st.session_state.pop("import_notice", None)
if import_notice is not None:
    notice_type, notice_message = import_notice
    if notice_type == "duplicate":
        st.sidebar.info(notice_message)
    else:
        st.sidebar.success(notice_message)

video_paths = discover_video_files(DATA_DIR)
if not video_paths:
    st.info(UI_TEXT_JA["no_videos"])
    st.stop()

selected_name = st.session_state.get("selected_video_name")
selected_index = next(
    (index for index, path in enumerate(video_paths) if path.name == selected_name),
    0,
)
original_video_path = st.sidebar.selectbox(
    UI_TEXT_JA["video_selector"],
    video_paths,
    index=selected_index,
    format_func=lambda path: path.name,
)
st.session_state["selected_video_name"] = original_video_path.name

selected_transcript = find_transcript_for_video(original_video_path, DATA_DIR)
transcript_data = None
transcript_error = None

if selected_transcript is not None:
    try:
        transcript_data, resolved_original_path = load_video_transcript(
            selected_transcript,
            DATA_DIR,
        )
        original_video_path = resolved_original_path
    except json.JSONDecodeError as error:
        transcript_error = UI_TEXT_JA["transcript_json_invalid"].format(error=error)
    except TranscriptFormatError as error:
        transcript_error = UI_TEXT_JA["transcript_structure_invalid"].format(
            error=error
        )
    except FileNotFoundError as error:
        transcript_error = UI_TEXT_JA["transcript_video_missing"].format(error=error)
    except (OSError, UnicodeError) as error:
        transcript_error = UI_TEXT_JA["transcript_read_failed"].format(error=error)

pending_auto_transcription_video = st.session_state.pop(
    "pending_auto_transcription_video",
    None,
)
should_auto_transcribe = (
    pending_auto_transcription_video == original_video_path.name
    and transcript_data is None
    and transcript_error is None
    and st.session_state.get("transcription_running_video")
    != original_video_path.name
)

segments = transcript_data["segments"] if transcript_data is not None else []
st.sidebar.caption(UI_TEXT_JA["segment_count"].format(count=len(segments)))
if transcript_data is not None and "duration" in transcript_data["video"]:
    st.sidebar.caption(
        UI_TEXT_JA["duration"].format(
            duration=format_timestamp(transcript_data["video"]["duration"])
        )
    )

playback_video_path = original_video_path
playback_assessment: PlaybackAssessment | None = None
playback_warning: str | None = None
video_id = original_video_path.stem
has_official_video_id = len(video_id) == 64 and all(
    character in "0123456789abcdef" for character in video_id
)

if has_official_video_id:
    try:
        playback_assessment = assess_playback(
            original_video_path,
            video_id,
            PLAYBACK_CACHE_DIR,
        )
        playback_video_path = playback_assessment.playback_path
        if playback_assessment.status == "compatibility_unknown":
            playback_warning = UI_TEXT_JA["playback_unknown"].format(
                video_codec=playback_assessment.media_info.video_codec,
                audio_codec=playback_assessment.media_info.audio_codec or "なし",
            )
    except PlaybackToolNotFoundError:
        playback_warning = UI_TEXT_JA["playback_tools_missing"]
    except MediaProbeError as error:
        playback_warning = UI_TEXT_JA["playback_probe_failed"].format(
            error=error
        )
else:
    playback_warning = UI_TEXT_JA["legacy_video"]

video_column, transcript_column = st.columns([1.15, 1], gap="large")

with video_column:
    st.subheader(UI_TEXT_JA["video_heading"])
    if playback_warning is not None:
        st.warning(playback_warning)
    if (
        playback_assessment is not None
        and playback_assessment.status == "proxy_required"
    ):
        st.warning(UI_TEXT_JA["proxy_required"])
        if st.button(
            UI_TEXT_JA["create_proxy_button"],
            key=f"create-playback-{video_id}",
            type="primary",
        ):
            try:
                with st.spinner(UI_TEXT_JA["creating_proxy"]):
                    ensure_playback_proxy(
                        original_video_path,
                        video_id,
                        PLAYBACK_CACHE_DIR,
                    )
            except PlaybackError as error:
                st.error(
                    UI_TEXT_JA["playback_create_failed"].format(error=error)
                )
            except OSError as error:
                st.error(
                    UI_TEXT_JA["playback_create_failed"].format(error=error)
                )
            else:
                st.session_state["playback_notice"] = (
                    UI_TEXT_JA["playback_create_success"]
                )
                st.rerun()

    playback_notice = st.session_state.pop("playback_notice", None)
    if playback_notice is not None:
        st.success(playback_notice)
    if playback_assessment is not None and playback_assessment.status == "proxy_cached":
        st.caption(UI_TEXT_JA["using_cached_proxy"])
    st.video(
        str(playback_video_path),
        format=video_mime_type(playback_video_path),
        alt=original_video_path.name,
    )
    st.caption(original_video_path.name)

with transcript_column:
    st.subheader(UI_TEXT_JA["transcript_heading"])
    transcription_notice = st.session_state.pop("transcription_notice", None)
    if transcription_notice is not None:
        st.success(transcription_notice)

    saved_error = st.session_state.get("transcription_error")
    if (
        isinstance(saved_error, tuple)
        and len(saved_error) == 2
        and saved_error[0] == original_video_path.name
    ):
        st.error(saved_error[1])

    if transcript_error is not None:
        st.error(transcript_error)
        if isinstance(selected_transcript, Path):
            st.info(UI_TEXT_JA["transcript_recovery"])
    elif transcript_data is None:
        st.info(UI_TEXT_JA["not_transcribed"])
        transcription_failed = (
            isinstance(saved_error, tuple)
            and len(saved_error) == 2
            and saved_error[0] == original_video_path.name
        )
        start_requested = should_auto_transcribe
        if not should_auto_transcribe:
            button_label = (
                UI_TEXT_JA["retry_transcription"]
                if transcription_failed
                else UI_TEXT_JA["start_transcription"]
            )
            start_requested = st.button(
                button_label,
                key=f"transcribe-{video_id}",
                type="primary",
                disabled=(
                    st.session_state.get("transcription_running_video")
                    is not None
                ),
            )

        if start_requested:
            try:
                run_transcription_with_status(original_video_path)
            except Exception as error:
                LOGGER.exception(
                    "Transcription failed for %s",
                    original_video_path,
                )
                st.session_state["transcription_error"] = (
                    original_video_path.name,
                    transcription_error_message(error),
                )
            else:
                st.session_state.pop("transcription_error", None)
                st.session_state["transcription_notice"] = UI_TEXT_JA[
                    "transcription_success"
                ]
            if should_auto_transcribe:
                st.session_state["import_notice"] = (
                    "success",
                    UI_TEXT_JA["import_success"],
                )
            st.rerun()
    else:
        if (
            isinstance(saved_error, tuple)
            and saved_error[0] == original_video_path.name
        ):
            st.session_state.pop("transcription_error", None)
        with st.container(height=650, border=True):
            for segment in segments:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                st.caption(f"{start} — {end}")
                st.write(segment["text"])
                st.divider()
