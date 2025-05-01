class Monologue:
    def __init__(self, speaker, start_time, end_time, duration, text, filepath=None):
        self.speaker = speaker
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.text = text
        self.filepath = filepath

    def __str__(self):
        return f"Speaker: {self.speaker}, Start Time: {self.start_time}, End Time: {self.end_time}, Duration: {self.duration}"


# Inherit from Monologue
class MonologueWithAudio(Monologue):
    # Since audio_data will be loaded using torchaudio.load(), the type of audio_data will be torch.Tensor
    def __init__(self, speaker, start_time, end_time, duration, text, filepath, audio_data, sample_rate):
        super().__init__(speaker, start_time, end_time, duration, text, filepath)
        self.audio_data = audio_data
        self.sample_rate = sample_rate


class ComparisonSummary:
    def __init__(self, speaker1_name, speaker2_name, correct_among_same_speaker_count, incorrect_among_same_speaker_count, correct_among_different_speaker_count, incorrect_among_different_speaker_count):
        self.speaker1_name = speaker1_name
        self.speaker2_name = speaker2_name
        self.correct_among_same_speaker_count = correct_among_same_speaker_count
        self.incorrect_among_same_speaker_count = incorrect_among_same_speaker_count
        self.correct_among_different_speaker_count = correct_among_different_speaker_count
        self.incorrect_among_different_speaker_count = incorrect_among_different_speaker_count


class ComparisonResult:
    def __init__(self, speaker1_name, speaker2_name, is_same_speaker, inference, basic_score, cosine_score, euclidean_score, integrated_score, filename1, filename2):
        self.speaker1_name = speaker1_name
        self.speaker2_name = speaker2_name
        self.is_same_speaker = is_same_speaker
        self.inference = inference
        self.basic_score = basic_score
        self.cosine_score = cosine_score
        self.euclidean_score = euclidean_score
        self.integrated_score = integrated_score
        self.filename1 = filename1
        self.filename2 = filename2
