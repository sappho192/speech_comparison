import torch
import torchaudio
from dtw import dtw

def compute_deltas(features, window_size=2):
    """
    デルタ特徴量の手動計算
    
    Args:
        features: 特徴量テンソル
        window_size: デルタ計算に使用する窓サイズ(デフォルト: 2)
    
    Returns:
        deltas: 計算されたデルタ特徴量
    """
    rows, cols = features.shape
    deltas = torch.zeros_like(features)
    # デルタ計算用の分母(正規化係数)
    denominator = 2 * sum([i*i for i in range(1, window_size+1)])
    # 各フレームに対してデルタを計算
    for i in range(rows):
        for t in range(-window_size, window_size+1):
            if 0 <= i+t < rows:
                # 現在のフレームからt離れたフレームとの重み付け計算
                weight = t
                deltas[i] += weight * features[i+t]
    # 正規化係数で割る
    deltas = deltas / denominator
    return deltas

def extract_optimized_mfccs(audio, sample_rate):
    """
    話者比較のための最適化されたMFCC特徴抽出
    
    Args:
        audio: 音声データ
        sample_rate: サンプリングレート
    
    Returns:
        combined_features: 結合されたMFCC、デルタ、デルタ・デルタ特徴量
    """
    # 話者比較のための最適化されたパラメータ設定
    mfcc = torchaudio.compliance.kaldi.mfcc(
        audio,
        num_ceps=20,                  # 話者特性をより詳細に捉えるため増加
        num_mel_bins=40,              # 周波数解像度の向上
        cepstral_lifter=27.0,         # 高次係数の強調
        frame_length=30.0,            # フレーム長の増加による安定した特徴の捉え
        frame_shift=10.0,             # 標準的なフレームシフトの維持
        preemphasis_coefficient=0.95,  # 話者の低周波特性の保持
        low_freq=50.0,                # 低周波の維持
        high_freq=8000.0,            # 話者識別に重要な中・高周波の包含
        sample_frequency=sample_rate,
        use_energy=True,              # エネルギー情報の含める
        remove_dc_offset=True,        # 環境差の軽減
        subtract_mean=True           # チャンネル効果の正規化
    )
    # デルタおよびデルタ・デルタ特徴量の計算(時間的変化情報の捉え)
    delta = compute_deltas(mfcc, window_size=2)
    delta_delta = compute_deltas(delta, window_size=2)
    # 全ての特徴を結合(縦方向に)
    combined_features = torch.cat([mfcc, delta, delta_delta], dim=1)
    return combined_features

def vad_simple(audio, sample_rate, threshold=0.01, frame_length=30, min_silence_duration=300):
    """
    シンプルな音声活動検出(VAD)による無音区間除去
    
    Args:
        audio: 音声データ
        sample_rate: サンプリングレート
        threshold: 閾値(デフォルト: 0.01)
        frame_length: フレーム長(ミリ秒、デフォルト: 30)
        min_silence_duration: 最小無音期間(ミリ秒、デフォルト: 300)
    
    Returns:
        活動音声区間のみを含む処理済み音声データ
    """
    frame_size = int(frame_length * sample_rate / 1000)
    hop_size = frame_size // 2
    
    # エネルギーに基づくVAD
    # unfoldが1次元テンソルを期待するため、チャンネル次元を削除
    if audio.dim() > 1:
        audio_1d = audio.squeeze(0)
    else:
        audio_1d = audio
        
    # フレーム分割が可能か確認
    if audio_1d.size(0) < frame_size:
        return audio  # 音声が短すぎる場合は元のデータを返す
        
    # エネルギー計算
    energy = []
    for i in range(0, audio_1d.size(0) - frame_size + 1, hop_size):
        frame = audio_1d[i:i+frame_size]
        energy.append(torch.mean(frame ** 2))
    energy = torch.tensor(energy)
    
    # 閾値に基づいて活動区間を特定
    is_speech = energy > threshold
    
    # 最小無音区間未満の場合、音声として扱う(小さな無音を無視)
    min_silence_frames = min_silence_duration * sample_rate / (1000 * hop_size)
    voiced_segments = []
    current_segment = []
    
    for i, active in enumerate(is_speech):
        if active:
            current_segment.append(i)
        elif len(current_segment) > 0:
            if len(current_segment) >= min_silence_frames:
                voiced_segments.append(current_segment)
            current_segment = []
    
    if len(current_segment) > 0:
        voiced_segments.append(current_segment)
        
    # 活動区間のオーディオ抽出
    if not voiced_segments:
        return audio  # 活動区間がない場合は元のデータを返す
        
    active_audio = []
    for segment in voiced_segments:
        start_frame = segment[0] * hop_size
        end_frame = (segment[-1] + 1) * hop_size + frame_size
        end_frame = min(end_frame, audio.size(-1))
        if audio.dim() > 1:
            active_audio.append(audio[:, start_frame:end_frame])
        else:
            active_audio.append(audio[start_frame:end_frame].unsqueeze(0))
            
    if not active_audio:
        return audio  # 活動区間がない場合は元のデータを返す
        
    return torch.cat(active_audio, dim=1)

def compare_audio_optimized(audio1, audio1_sample_rate, audio2, audio2_sample_rate, use_vad=True, distance_method="euclidean"):
    """
    最適化された話者比較関数
    
    Args:
        audio1: 1つ目の音声データ
        audio2: 2つ目の音声データ
        sample_rate: サンプリングレート
        use_vad: VADを使用するかどうか(デフォルト: True)
        distance_method: 距離測定方法(デフォルト: "euclidean")
    
    Returns:
        similarity_score: 相似度スコア(距離が小さいほど相似度は高い)
    """
    # 1. VAD適用(オプション)
    if use_vad:
        audio1 = vad_simple(audio1, audio1_sample_rate)
        audio2 = vad_simple(audio2, audio2_sample_rate)
    
    # 2. 最適化されたMFCC特徴抽出
    features1 = extract_optimized_mfccs(audio1, audio1_sample_rate)
    features2 = extract_optimized_mfccs(audio2, audio2_sample_rate)
    
    # 3. 特徴量正規化(Zスコア)
    # NaNを防ぐため小さな値を追加し、0での除算を防止
    features1 = (features1 - features1.mean(dim=0)) / (features1.std(dim=0) + 1e-6)
    features2 = (features2 - features2.mean(dim=0)) / (features2.std(dim=0) + 1e-6)
    
    # NaN値処理
    features1 = torch.nan_to_num(features1)
    features2 = torch.nan_to_num(features2)
    
    # 4. 距離測定方法の選択
    if distance_method == "dtw":
        # DTW距離計算
        features1_np = features1.numpy()
        features2_np = features2.numpy()
        dtw_result = dtw(features1_np, features2_np, dist_method="euclidean", distance_only=True)
        distance = dtw_result.normalizedDistance
    elif distance_method == "cosine":
        # コサイン距離計算
        features1_mean = features1.mean(dim=0)
        features2_mean = features2.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            features1_mean.unsqueeze(0),
            features2_mean.unsqueeze(0)
        )
        distance = 1 - cos_sim.item()
    else:  # デフォルトのユークリッド距離
        # ユークリッド距離計算(各フレームのペアの距離の平均)
        # 長さを揃えるため短い方をパディング
        max_length = max(features1.shape[0], features2.shape[0])
        if features1.shape[0] < max_length:
            features1 = torch.cat([features1, torch.zeros(max_length - features1.shape[0], features1.shape[1])])
        elif features2.shape[0] < max_length:
            features2 = torch.cat([features2, torch.zeros(max_length - features2.shape[0], features2.shape[1])])
        distance = torch.sqrt(torch.sum((features1 - features2)**2)) / max_length
    
    # 相似度スコア計算(距離が小さいほど相似度は高い)
    similarity_score = 1 / (1 + distance)
    return similarity_score

def compare_audio_basic(audio1, audio1_sample_rate, audio2, audio2_sample_rate):
    """
    基本的な音声比較関数(比較用に保持)
    
    Args:
        audio1: 1つ目の音声データ
        audio2: 2つ目の音声データ
        sample_rate: サンプリングレート
    
    Returns:
        similarity_score: 相似度スコア
    """
    max_length = max(audio1.size(1), audio2.size(1))
    if audio1.size(1) < max_length:
        audio1 = torch.cat((audio1, torch.zeros(1, max_length - audio1.size(1))), dim=1)
    elif audio2.size(1) < max_length:
        audio2 = torch.cat((audio2, torch.zeros(1, max_length - audio2.size(1))), dim=1)
    mfcc1 = torchaudio.compliance.kaldi.mfcc(audio1, num_ceps=12, sample_frequency=audio1_sample_rate)
    mfcc2 = torchaudio.compliance.kaldi.mfcc(audio2, num_ceps=12, sample_frequency=audio2_sample_rate)
    max_mfcc_length = max(mfcc1.shape[0], mfcc2.shape[0])
    if mfcc1.shape[0] < max_mfcc_length:
        mfcc1 = torch.cat((mfcc1, torch.zeros(max_mfcc_length - mfcc1.shape[0], mfcc1.shape[1])))
    elif mfcc2.shape[0] < max_mfcc_length:
        mfcc2 = torch.cat((mfcc2, torch.zeros(max_mfcc_length - mfcc2.shape[0], mfcc2.shape[1])))
    distance = torch.sqrt(torch.sum((mfcc1 - mfcc2)**2))
    max_distance = torch.sqrt(torch.sum((mfcc1 + mfcc2)**2))
    similarity_score = 1 - (distance / max_distance)
    return similarity_score

def compare_audio_advanced(audio1, audio2, sample_rate):
    """
    高度な音声比較関数(比較用に保持)
    
    Args:
        audio1: 1つ目の音声データ
        audio2: 2つ目の音声データ
        sample_rate: サンプリングレート
    
    Returns:
        similarity_score: 相似度スコア
    """
    mfcc1 = torchaudio.compliance.kaldi.mfcc(audio1, num_ceps=12, sample_frequency=sample_rate)
    mfcc2 = torchaudio.compliance.kaldi.mfcc(audio2, num_ceps=12, sample_frequency=sample_rate)
    mfcc1_np = mfcc1.numpy()
    mfcc2_np = mfcc2.numpy()
    # 特徴量正規化
    mfcc1_np = (mfcc1_np - mfcc1_np.mean(axis=0)) / (mfcc1_np.std(axis=0) + 1e-6)
    mfcc2_np = (mfcc2_np - mfcc2_np.mean(axis=0)) / (mfcc2_np.std(axis=0) + 1e-6)
    # DTW距離計算
    dtw_result = dtw(mfcc1_np, mfcc2_np, dist_method="euclidean", distance_only=True)
    similarity_score = 1 - dtw_result.normalizedDistance
    return similarity_score

def integrate_scores(basic_score, euclidean_score, cosine_score, 
                     basic_threshold=0.23, euclidean_threshold=0.65, cosine_threshold=0.5,
                     basic_weight=0.4, euclidean_weight=0.2, cosine_weight=0.4):
    """
    複数のスコアを統合して最終的な判断を行う
    
    Args:
        basic_score: 基本方法のスコア
        euclidean_score: ユークリッド距離法のスコア
        cosine_score: コサイン距離法のスコア
        basic_threshold: 基本方法の閾値
        euclidean_threshold: ユークリッド距離法の閾値
        cosine_threshold: コサイン距離法の閾値
    
    Returns:
        tuple: (統合スコア, 判断結果)
    """
    # スコアの正規化（0-1の範囲に収める）
    normalized_scores = {
        'basic': max(0.0, min(1.0, basic_score)),
        'euclidean': max(0.0, min(1.0, euclidean_score)),
        'cosine': max(0.0, min(1.0, cosine_score))
    }
    
    # 各方法の判断結果（個別の閾値を使用）
    decisions = {
        'basic': normalized_scores['basic'] >= basic_threshold,
        'euclidean': normalized_scores['euclidean'] >= euclidean_threshold,
        'cosine': normalized_scores['cosine'] >= cosine_threshold
    }
    
    # 統合スコアの計算（加重平均）
    weights = {
        'basic': basic_weight,      # 基本方法
        'euclidean': euclidean_weight,  # ユークリッド距離法
        'cosine': cosine_weight      # コサイン距離法
    }
    
    integrated_score = sum(
        weights[method] * normalized_scores[method] 
        for method in weights
    )
    
    # 多数決による最終判断
    final_decision = sum(decisions.values()) >= len(decisions) / 2
    
    return integrated_score, final_decision