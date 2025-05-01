import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pickle
from itertools import combinations
from typing import Callable, Tuple, List
import multiprocessing as mp
from functools import partial
import torch
from tqdm import tqdm
from multiprocessing import Pool
from loguru import logger

class DistanceMetric:
    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """코사인 거리 계산 (1 - 코사인 유사도)"""
        return 1 - cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """유클리드 거리 계산"""
        return euclidean(emb1.flatten(), emb2.flatten())

def process_pair(pair_data: Tuple, embeddings: dict, distance_fn: Callable) -> Tuple[float, bool]:
    """단일 쌍에 대한 거리 계산"""
    file1, file2, is_same_speaker = pair_data
    
    # 파일명에서 '.wav' 제거
    key1 = file1.replace('.wav', '')
    key2 = file2.replace('.wav', '')
    
    if key1 in embeddings and key2 in embeddings:
        emb1 = embeddings[key1]
        emb2 = embeddings[key2]
        score = distance_fn(emb1, emb2)
        return score, is_same_speaker
    return None

def classify_speaker_cosine(emb1: np.ndarray, emb2: np.ndarray, threshold: float) -> bool:
    """임베딩 간의 거리를 기반으로 화자 분류"""
    distance = DistanceMetric.cosine_distance(emb1, emb2)
    logger.info(f"코사인 거리: {distance}")
    return distance <= threshold


def create_different_speaker_pairs(speaker_group: List[str], other_speakers: List[str], 
                                 labels_df: pd.DataFrame, pairs_per_speaker: int) -> List[Tuple]:
    """특정 화자 그룹에 대해 다른 화자와의 쌍을 효율적으로 생성"""
    different_speaker_pairs = []
    
    # 다른 화자들의 파일 미리 필터링
    other_speakers_files = labels_df[labels_df['SpkrID'].isin(other_speakers)]
    
    for spkr1 in speaker_group:
        # 현재 화자의 모든 파일
        spkr1_files = labels_df[labels_df['SpkrID'] == spkr1]['FileName'].values
        
        # 필요한 쌍보다 더 많은 수의 샘플 생성 (중복 제거를 고려)
        oversample_factor = 2
        n_samples = pairs_per_speaker * oversample_factor
        
        # 벌크 샘플링 수행
        file1_samples = np.random.choice(spkr1_files, size=n_samples, replace=True)
        file2_samples = np.random.choice(other_speakers_files['FileName'].values, size=n_samples, replace=True)
        
        # 쌍 생성 및 중복 제거
        pairs = set()
        for f1, f2 in zip(file1_samples, file2_samples):
            if len(pairs) >= pairs_per_speaker:
                break
            pair = (f1, f2, False)
            pairs.add(pair)
        
        different_speaker_pairs.extend(list(pairs)[:pairs_per_speaker])
    
    return different_speaker_pairs

def execute_task(task):
    """병렬 처리를 위한 태스크 실행 함수"""
    return task()

def load_embeddings_to_gpu(embeddings: dict) -> Tuple[dict, torch.Tensor, torch.device]:
    """임베딩을 GPU 메모리에 로드 (필요 시 불필요한 차원 제거)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모든 임베딩을 하나의 텐서로 변환 (단, 불필요한 차원은 제거)
    embedding_keys = list(embeddings.keys())
    embedding_values = [embeddings[k] for k in embedding_keys]
    
    # 스택 결과가 (N,1,192)라면 squeeze 수행
    stacked = np.stack(embedding_values)
    if len(stacked.shape) == 3 and stacked.shape[1] == 1:
        stacked = np.squeeze(stacked, axis=1)
    
    embeddings_tensor = torch.tensor(stacked, device=device)
    
    # 키를 인덱스로 매핑하는 딕셔너리 생성
    key_to_idx = {k: i for i, k in enumerate(embedding_keys)}
    
    # logger.info(f"임베딩 텐서 크기: {embeddings_tensor.shape}")
    # logger.info(f"GPU 메모리에 로드된 임베딩 수: {len(key_to_idx)}")
    
    return key_to_idx, embeddings_tensor, device

def process_batch_gpu(batch_pairs: List[Tuple], key_to_idx: dict, embeddings_tensor: torch.Tensor, 
                     device: torch.device, distance_fn: Callable) -> List[Tuple[float, bool]]:
    """GPU를 사용하여 배치 단위로 거리 계산"""
    # 배치의 임베딩 인덱스를 수집
    idx1_list = []
    idx2_list = []
    is_same_list = []
    
    for file1, file2, is_same_speaker in batch_pairs:
        key1 = file1.replace('.wav', '')
        key2 = file2.replace('.wav', '')
        
        if key1 in key_to_idx and key2 in key_to_idx:
            idx1_list.append(key_to_idx[key1])
            idx2_list.append(key_to_idx[key2])
            is_same_list.append(is_same_speaker)
    
    if not idx1_list:
        return []
    
    # 인덱스를 사용하여 GPU 메모리에서 임베딩 가져오기
    idx1_tensor = torch.tensor(idx1_list, device=device)
    idx2_tensor = torch.tensor(idx2_list, device=device)
    
    emb1_tensor = embeddings_tensor[idx1_tensor]
    emb2_tensor = embeddings_tensor[idx2_tensor]
    
    # GPU에서 거리 계산
    with torch.no_grad():
        if distance_fn == DistanceMetric.cosine_distance:
            # 각 차원별 곱을 계산
            dot_product = torch.sum(emb1_tensor * emb2_tensor, dim=1)  # (batch_size,)
            # 각 벡터의 norm 계산
            norm1 = torch.norm(emb1_tensor, dim=1)  # (batch_size,)
            norm2 = torch.norm(emb2_tensor, dim=1)  # (batch_size,)
            # 코사인 유사도 계산
            scores = 1 - (dot_product / (norm1 * norm2))  # (batch_size,)
        else:
            # 벡터 차이 계산
            diff = emb1_tensor - emb2_tensor  # (batch_size, 192)
            # 유클리드 거리 계산
            scores = torch.sqrt(torch.sum(diff * diff, dim=1))  # (batch_size,)
        
    scores = scores.cpu().numpy()
        
    
    return list(zip(scores, is_same_list))

def compute_scores(embeddings: dict, labels_df: pd.DataFrame, 
                  distance_fn: Callable, random_seed: int = 42) -> Tuple[List[float], List[bool]]:
    """같은 화자 쌍과 다른 화자 쌍에 대해 거리 점수 계산 (개선된 병렬 처리)"""
    np.random.seed(random_seed)
    
    # SpkrID별로 파일 그룹화
    speaker_files = {}
    for _, row in labels_df.iterrows():
        spkr_id = row['SpkrID']
        file_name = row['FileName']
        if spkr_id not in speaker_files:
            speaker_files[spkr_id] = []
        speaker_files[spkr_id].append(file_name)
    
    # 같은 화자 쌍 생성
    same_speaker_pairs = []
    for spkr_files in speaker_files.values():
        if len(spkr_files) >= 2:
            pairs = list(combinations(spkr_files, 2))
            same_speaker_pairs.extend([(f1, f2, True) for f1, f2 in pairs])
    
    # 전체 화자 목록과 필요한 쌍의 수 계산
    all_speakers = list(speaker_files.keys())
    n_same_pairs = len(same_speaker_pairs)
    pairs_per_speaker = (n_same_pairs // len(all_speakers)) + 1
    
    # 화자들을 12개의 그룹으로 나누기
    n_groups = 12
    speakers_per_group = len(all_speakers) // n_groups
    remainder = len(all_speakers) % n_groups
    
    speaker_groups = []
    start_idx = 0
    for i in range(n_groups):
        group_size = speakers_per_group + (1 if i < remainder else 0)
        speaker_groups.append(all_speakers[start_idx:start_idx + group_size])
        start_idx += group_size
    
    # 병렬 처리로 다른 화자 쌍 생성
    pool = mp.Pool(n_groups)
    different_pairs_tasks = []
    
    for speaker_group in speaker_groups:
        other_speakers = [s for s in all_speakers if s not in speaker_group]
        task = partial(create_different_speaker_pairs, 
                      speaker_group, other_speakers, labels_df, pairs_per_speaker)
        different_pairs_tasks.append(task)
    
    logger.info("다른 화자 쌍 생성 중...")
    different_speaker_pairs = []
    for result in pool.imap_unordered(execute_task, different_pairs_tasks):
        different_speaker_pairs.extend(result)
    
    # 모든 쌍 합치기
    all_pairs = same_speaker_pairs + different_speaker_pairs
    
    logger.info(f"생성된 총 쌍의 수: {len(all_pairs)}")  # 디버그 출력
    logger.info(f"첫 번째 쌍 예시: {all_pairs[0]}")  # 디버그 출력
    
    # GPU에 임베딩 로드
    logger.info("GPU에 임베딩 로드 중...")
    key_to_idx, embeddings_tensor, device = load_embeddings_to_gpu(embeddings)
    
    # GPU 배치 처리
    batch_size = 8192*16
    all_results = []
    
    logger.info(f"\nGPU를 사용하여 거리 계산 중...")
    for i in tqdm(range(0, len(all_pairs), batch_size)):
        batch_pairs = all_pairs[i:i + batch_size]
        batch_results = process_batch_gpu(batch_pairs, key_to_idx, embeddings_tensor, 
                                        device, distance_fn)
        all_results.extend(batch_results)
    
    logger.info(f"총 결과 수: {len(all_results)}")  # 디버그 출력
    
    # None이 아닌 결과만 필터링
    valid_results = [r for r in all_results if r is not None]
    logger.info(f"유효한 결과 수: {len(valid_results)}")  # 디버그 출력
    scores, labels = zip(*valid_results)
    
    #logger.info(f"총 쌍의 수: {len(scores)}")
    #logger.info(f"같은 화자 쌍의 수: {sum(labels)}")
    #logger.info(f"다른 화자 쌍의 수: {len(labels) - sum(labels)}")
    
    return list(scores), list(labels)

# 전역 변수 (각 프로세스에서 사용될 scores, labels, score_chunk_size)
global_scores_arr = None
global_labels_arr = None
global_score_chunk_size = None

def init_worker(scores, labels, score_chunk_size):
    """
    각 워커 프로세스의 전역 변수를 초기화합니다.
    """
    global global_scores_arr, global_labels_arr, global_score_chunk_size
    global_scores_arr = scores
    global_labels_arr = labels
    global_score_chunk_size = score_chunk_size

def worker_eer(args):
    """
    각 프로세스에서 임계값 청크에 대해 false_accept, false_reject 값을 계산합니다.
    """
    import numpy as np
    start_idx, thresholds_chunk = args
    false_accept_chunk = np.zeros(len(thresholds_chunk), dtype=np.int64)
    false_reject_chunk = np.zeros(len(thresholds_chunk), dtype=np.int64)
    
    # 전역 변수(global_scores_arr, global_labels_arr)를 score_chunk_size 단위로 처리
    for j in range(0, len(global_scores_arr), global_score_chunk_size):
        scores_sub = global_scores_arr[j:j + global_score_chunk_size]
        labels_sub = global_labels_arr[j:j + global_score_chunk_size]
        
        # 현재 서브 배치와 thresholds_chunk에 대해 예측값 계산 (브로드캐스팅 사용)
        predictions = scores_sub[:, None] <= thresholds_chunk[None, :]  # shape: (sub_batch, len(thresholds_chunk))
        false_accept_chunk += np.sum(predictions & (~labels_sub[:, None]), axis=0)
        false_reject_chunk += np.sum((~predictions) & (labels_sub[:, None]), axis=0)
    
    return start_idx, false_accept_chunk, false_reject_chunk

def find_eer(scores: List[float], labels: List[bool],distance_metric: str,
             num_iterations: int = 5,
             num_cores: int = 16,
             score_chunk_size: int = 100000) -> Tuple[float, float]:
    """
    멀티프로세싱을 활용하여 개선된 임계값 검색 방식을 통해 EER과 최적 임계값을 계산합니다.
    
    초기 임계값 범위는 distance_metric에 따라 설정되며,
      - 코사인 유사도일 경우: -1부터 1까지,
      - 유클리드 거리일 경우: 0부터 65535까지.
      
    이후 지정된 코어 수(num_cores, 기본 16)를 활용하여 해당 범위를 균등 분할한 후보 임계값들에 대해
    FAR과 FRR를 계산하고, 이 둘의 차이가 최소가 되는 후보 임계값 주변 범위로 재설정하는 과정을
    num_iterations (기본 5회) 반복합니다.
    
    최종적으로 해당 임계값에서 FAR과 FRR를 기반으로 EER을 산출하며, (EER, 최적 임계값)을 반환합니다.
    """
    import numpy as np
    from multiprocessing import Pool

    # numpy 배열로 변환 및 동일/다른 화자 샘플 수 확인
    scores_arr = np.array(scores)
    labels_arr = np.array(labels, dtype=bool)
    
    n_same = labels_arr.sum()
    n_diff = (~labels_arr).sum()
    
    if n_same == 0 or n_diff == 0:
        raise ValueError("동일한 화자 또는 다른 화자의 샘플 수가 충분하지 않습니다.")
        
    # distance_metric에 따라 초기 임계값 범위 설정
    if distance_metric == "cosine":
        low, high = -1.0, 1.0
    elif distance_metric == "euclidean":
        # 유클리드 거리는 음수가 나올 수 없으므로 초기 범위의 하한을 0으로 설정해야 합니다.
        # 고정 값으로 사용할 경우:
        low, high = 0.0, 65535.0
        # 또는 아래와 같이 동적으로 범위를 설정할 수 있습니다.
        # low, high = scores_arr.min(), scores_arr.max()
    else:
        raise ValueError(f"지원되지 않는 distance_metric: {distance_metric}")
    
    # 전역 변수 초기화 (각 워커에서 사용)
    global global_scores_arr, global_labels_arr, global_score_chunk_size
    global_scores_arr = scores_arr
    global_labels_arr = labels_arr
    global_score_chunk_size = score_chunk_size
    
    best_threshold = None
    pool = Pool(processes=num_cores, initializer=init_worker, initargs=(scores_arr, labels_arr, score_chunk_size))
    
    try:
        for iteration in range(num_iterations):
            # 현재 범위 내에서 num_cores 개의 후보 임계값을 균등 분할하여 생성
            candidate_thresholds = np.linspace(low, high, num_cores)
            
            # 각 워커에 할당할 임계값 청크로 분할 (각 청크는 후보 임계값의 부분 배열)
            threshold_chunks = np.array_split(candidate_thresholds, num_cores)
            chunks = []
            start_idx = 0
            for chunk in threshold_chunks:
                chunks.append((start_idx, chunk))
                start_idx += len(chunk)
                
            # 각 워커에서 주어진 임계값 청크에 대한 false accept, false reject 계산
            results = pool.map(worker_eer, chunks)
            
            # 각 청크의 결과를 병합
            num_candidates = candidate_thresholds.shape[0]
            total_false_accept = np.zeros(num_candidates, dtype=np.int64)
            total_false_reject = np.zeros(num_candidates, dtype=np.int64)
            for start_idx, fa_chunk, fr_chunk in results:
                total_false_accept[start_idx:start_idx+len(fa_chunk)] = fa_chunk
                total_false_reject[start_idx:start_idx+len(fr_chunk)] = fr_chunk
            
            # FAR과 FRR 계산
            far = total_false_accept / (n_diff if n_diff > 0 else 1)
            frr = total_false_reject / (n_same if n_same > 0 else 1)
            diffs = np.abs(far - frr)
            
            # FAR과 FRR 차이가 최소인 후보 임계값 선택
            best_idx = np.argmin(diffs)
            best_threshold = candidate_thresholds[best_idx]
            
            # 다음 반복을 위한 범위 재설정: 선택된 임계값 주변 값으로 경계를 설정
            if best_idx == 0:
                new_low = candidate_thresholds[0]
                new_high = candidate_thresholds[1] if num_candidates > 1 else candidate_thresholds[0]
            elif best_idx == num_candidates - 1:
                new_low = candidate_thresholds[-2]
                new_high = candidate_thresholds[-1]
            else:
                new_low = candidate_thresholds[best_idx - 1]
                new_high = candidate_thresholds[best_idx + 1]
            
            low, high = new_low, new_high
        # end for
    finally:
        pool.close()
        pool.join()
    
    # 최종 best_threshold에서 FAR과 FRR를 재계산하고, EER 산출
    predictions = scores_arr <= best_threshold
    false_accept = np.sum(predictions & (~labels_arr))
    false_reject = np.sum((~predictions) & (labels_arr))
    far_final = false_accept / (n_diff if n_diff > 0 else 1)
    frr_final = false_reject / (n_same if n_same > 0 else 1)
    eer = (far_final + frr_final) / 2.0
    
    return eer, best_threshold 