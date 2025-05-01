import json
from pathlib import Path
from typing import List, Dict

from loguru import logger

import torch
import torchaudio

from speech_comparison.speaker import Monologue, MonologueWithAudio, ComparisonSummary

def get_time_info(filepath: Path):
    # filepath is like: post_dataset/train/pure/0bb642e7-d702-426b-92f3-8ac1e3b4e77d/1.488_10.612.flac
    # We will get filename and split it to get start_time, end_time
    # Then we will get duration from start_time and end_time
    filename = filepath.stem
    start_time, end_time = filename.split("_")[0], filename.split("_")[1]
    duration = float(end_time) - float(start_time)

    return start_time, end_time, duration


def load_pure_monologue_metadata(pure_subsplit_path: Path):
    pure_monologue_metadata = []
    metadata_filename = "pure_monologue_metadata.jsonl"

    with open(pure_subsplit_path.joinpath(metadata_filename), "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line.strip())
            speaker = json_data["speaker"]
            text = json_data["text"]
            filepath = json_data["filepath"]
            start_time, end_time, duration = get_time_info(Path(filepath))
            pure_monologue_metadata.append(Monologue(speaker, start_time, end_time, duration, text, filepath))

    return pure_monologue_metadata


def load_monologue_audio(dataset_root: Path, speaker: str, monologues: List[Monologue]) -> List[MonologueWithAudio]:
    monologue_audio = []
    for monologue in monologues:
        audio_path = validate_audio_path(dataset_root, Path(monologue.filepath))
        audio, sample_rate = torchaudio.load(audio_path)
        monologue_audio.append(MonologueWithAudio(speaker, monologue.start_time, monologue.end_time, monologue.duration, monologue.text, monologue.filepath, audio, sample_rate))

    return monologue_audio

def generate_speaker_database(dataset_root: Path, device: torch.device, pure_monologue_metadata: List[Monologue]) -> Dict[str, List[MonologueWithAudio]]:
    speaker_database = dict()

    if "cuda" in device.type:
        logger.info(f"Using {device} when generating speaker database")

    for monologue in pure_monologue_metadata:
        if monologue.speaker not in speaker_database:
            speaker_database[monologue.speaker] = []
        audio_path = validate_audio_path(dataset_root, Path(monologue.filepath))
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.unsqueeze(0).to(device)
        monologue_with_audio = MonologueWithAudio(monologue.speaker, monologue.start_time, monologue.end_time, monologue.duration, monologue.text, monologue.filepath, audio, sample_rate)
        speaker_database[monologue.speaker].append(monologue_with_audio)

    return speaker_database



def print_comparison_summary(summary: ComparisonSummary):
    logger.info(f"Speaker 1: {summary.speaker1_name}")
    logger.info(f"Speaker 2: {summary.speaker2_name}")
    logger.info(f"Correct among same speaker: {summary.correct_among_same_speaker_count}")
    logger.info(f"Incorrect among same speaker: {summary.incorrect_among_same_speaker_count}")
    logger.info(f"Correct among different speaker: {summary.correct_among_different_speaker_count}")
    logger.info(f"Incorrect among different speaker: {summary.incorrect_among_different_speaker_count}")
    logger.info(f"Accuracy among same speaker: {summary.correct_among_same_speaker_count / (summary.correct_among_same_speaker_count + summary.incorrect_among_same_speaker_count)}")
    logger.info(f"Accuracy among different speaker: {summary.correct_among_different_speaker_count / (summary.correct_among_different_speaker_count + summary.incorrect_among_different_speaker_count)}")
    correct = summary.correct_among_same_speaker_count + summary.correct_among_different_speaker_count
    incorrect = summary.incorrect_among_same_speaker_count + summary.incorrect_among_different_speaker_count
    recall = correct / (correct + incorrect)
    precision = correct / (correct + summary.incorrect_among_same_speaker_count)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    logger.info(f"Total F1 score: {f1_score}")


def validate_audio_path(dataset_root: Path, audio_path: Path):
    if audio_path.exists():
        return audio_path
    filename_with_extension = audio_path.parts[-1]
    job_id = audio_path.parts[-2]
    subsplit_name = audio_path.parts[-3]
    split_name = audio_path.parts[-4]
    audio_path = dataset_root.joinpath(split_name, subsplit_name, job_id, filename_with_extension)
    if audio_path.exists():
        return audio_path
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

