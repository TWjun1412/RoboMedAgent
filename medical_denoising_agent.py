"""
RoboMedAgent:Framework for Medical Dialogue Denoising
Detector-Editor-Arbiter Pipeline

Framework Overview:
1. Detector: Identifies potential edit signals with standardized tags
2. Editor: Processes deterministic and candidate edits with scoring
3. Arbiter: LLM-based final decision making and conflict resolution
4. Evaluation: Multi-dimensional quality assessment

Output format for edits:
[{
  "start_char": int,
  "end_char": int,
  "op": "REPLACE" | "DELETE" | "INSERT",
  "cand_texts": ["候选1", "候选2", ...],
  "score": float,
  "tag": "RPT|SPL|GRM|AMB|NOS",
  "edit_type": "deterministic|candidate"
}, ...]
"""

from typing import List, Dict, Tuple, Optional, Callable
import re, json, itertools
from dataclasses import dataclass, asdict
from collections import Counter
import time
from functools import wraps

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import symspellpy
from symspellpy import SymSpell, Verbosity
try:
    from openai import APIConnectionError, APIError, RateLimitError
except ImportError:
    # 兼容旧版本
    APIConnectionError = Exception
    APIError = Exception
    RateLimitError = Exception

# ==================== 重试工具函数 ====================
def retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    API调用重试装饰器
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        backoff_factor: 退避因子
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (APIConnectionError, RateLimitError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = initial_delay * (backoff_factor ** attempt)
                        time.sleep(delay)
                        continue
                    else:
                        raise
                except APIError as e:
                    # API错误通常不需要重试
                    raise
                except Exception as e:
                    # 其他错误，根据情况决定是否重试
                    if "Connection" in str(e) or "timeout" in str(e).lower():
                        last_exception = e
                        if attempt < max_retries - 1:
                            delay = initial_delay * (backoff_factor ** attempt)
                            time.sleep(delay)
                            continue
                    raise
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
from symspellpy import SymSpell, Verbosity
import nltk
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"⚠️ NLTK punkt数据包下载失败: {e}")
    print("💡 这不会影响主要功能，可以继续使用")

# ---------- Helper: normalize edits ----------
def normalize_edits(edits):
    """
    Accepts a list of SpanEdit or dict-like edits and returns a list of dicts with keys:
    'start_char', 'end_char', 'op', 'cand_texts', 'score'.
    """
    norm = []
    for e in edits:
        if isinstance(e, SpanEdit):
            d = e.to_dict()
        elif isinstance(e, dict):
            d = e.copy()
        else:
            try:
                d = {
                    "start_char": getattr(e, "start_char"),
                    "end_char": getattr(e, "end_char"),
                    "op": getattr(e, "op"),
                    "cand_texts": getattr(e, "cand_texts"),
                    "score": getattr(e, "score", 0.0)
                }
            except Exception:
                continue

        if "start" in d and "start_char" not in d:
            d["start_char"] = d.pop("start")
        if "end" in d and "end_char" not in d:
            d["end_char"] = d.pop("end")

        d.setdefault("start_char", 0)
        d.setdefault("end_char", d["start_char"])
        d.setdefault("op", "REPLACE")
        d.setdefault("cand_texts", [""])
        d.setdefault("score", 0.0)

        norm.append(d)
    return norm


# --------------------------
# Data structure
# --------------------------
@dataclass
class SpanEdit:
    start_char: int
    end_char: int
    op: str  # "REPLACE"/"DELETE"/"INSERT"
    cand_texts: List[str]
    score: float
    tag: str = ""  # "RPT|SPL|GRM|AMB|NOS"
    edit_type: str = "deterministic"  # "deterministic|candidate"
    detector_name: str = ""  # 来源检测器名称

    def to_dict(self):
        return asdict(self)

class BaseExtractorModule:
    def detect(self, text: str) -> List[SpanEdit]:
        raise NotImplementedError()

# --------------------------
# 1) GEC / seq2edit
# --------------------------
import re
import difflib
from typing import List

class GECTagger(BaseExtractorModule):
    def __init__(self, model_name_or_path=r"C:\Users\Laptop4\Desktop\models\grammar_error_correcter_v1"):
        # 使用本地路径加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def detect(self, text: str) -> List[SpanEdit]:
        inputs = self.tokenizer([text], max_length=128, truncation=True, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=128)
        corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        edits = []
        if corrected != text:
            orig_tokens = text.split()
            corr_tokens = corrected.split()

            # 使用 difflib 对齐
            diff = difflib.SequenceMatcher(None, orig_tokens, corr_tokens)

            for tag, i1, i2, j1, j2 in diff.get_opcodes():
                if tag == "replace":
                    span = re.search(re.escape(" ".join(orig_tokens[i1:i2])), text)
                    if span:
                        edits.append(SpanEdit(
                            span.start(),
                            span.end(),
                            "REPLACE",
                            [" ".join(corr_tokens[j1:j2])],
                            0.9,
                            tag="GRM",
                            edit_type="deterministic",
                            detector_name="GEC"
                        ))
                elif tag == "insert":
                    # 插入位置：原始 token 的结尾
                    pos = (re.search(re.escape(" ".join(orig_tokens[:i1])), text).end()
                           if i1 > 0 else 0)
                    edits.append(SpanEdit(
                        pos,
                        pos,
                        "INSERT",
                        [" ".join(corr_tokens[j1:j2])],
                        0.8,
                        tag="GRM",
                        edit_type="deterministic",
                        detector_name="GEC"
                    ))
                elif tag == "delete":
                    span = re.search(re.escape(" ".join(orig_tokens[i1:i2])), text)
                    if span:
                        edits.append(SpanEdit(
                            span.start(),
                            span.end(),
                            "DELETE",
                            [""],
                            0.8,
                            tag="GRM",
                            edit_type="deterministic",
                            detector_name="GEC"
                        ))

        return edits

# --------------------------
# 2) SpellChecker + medical guard
# --------------------------
class SpellChecker(BaseExtractorModule):
    def __init__(self, medical_terms_manager=None):
        self.medical_terms_manager = medical_terms_manager
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        # 可以加载大词典，如 sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

    def is_medical_term(self, token: str) -> bool:
        if self.medical_terms_manager:
            return self.medical_terms_manager.is_medical_term(token)
        return False

    def detect(self, text: str) -> List[SpanEdit]:
        edits = []
        for m in re.finditer(r"\b[A-Za-z0-9'-]+\b", text):
            token = m.group(0)
            if self.is_medical_term(token):
                continue
            suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != token:
                edits.append(SpanEdit(
                    start_char=m.start(),
                    end_char=m.end(),
                    op="REPLACE",
                    cand_texts=[s.term for s in suggestions[:3]],
                    score=1 - suggestions[0].distance/3,
                    tag="SPL",
                    edit_type="deterministic",
                    detector_name="SpellChecker"
                ))
        return edits

# --------------------------
# 3) Repetition detector
# --------------------------
class RepetitionDetector(BaseExtractorModule):
    def __init__(self, max_ngram=3):
        self.max_ngram = max_ngram

    def detect(self, text: str) -> List[SpanEdit]:
        edits = []
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return edits

        char_positions = []
        idx = 0
        for t in tokens:
            start = text.find(t, idx)
            end = start + len(t)
            char_positions.append((t, start, end))
            idx = end

        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - 2*n + 1):
                a = tokens[i:i+n]
                b = tokens[i+n:i+2*n]
                if a == b:
                    start = char_positions[i+n][1]
                    end = char_positions[i+2*n-1][2]
                    edits.append(SpanEdit(
                        start, end, "DELETE", [""], 0.9,
                        tag="RPT", edit_type="deterministic", detector_name="Repetition"
                    ))
        return edits

# --------------------------
# 4) WSD / Ambiguity detector (GlossBERT-based)
# --------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import List, Tuple

class CombinedMedicalDetector(BaseExtractorModule):
    """
    Combined LLM-based detector for both ambiguity detection and non-medical dialogue detection.
    """

    def __init__(self, api_key: str = None, model_name="gpt-4o-mini"):
        super().__init__()
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.chatanywhere.tech/v1",
                timeout=60.0  # 设置60秒超时
            )
            self.model_name = model_name
        else:
            self.client = None
            self.model_name = model_name

    def _detect_with_llm(self, text: str) -> str:
        """使用LLM同时检测歧义词和非医患对话"""
        if not self.client:
            return text
            
        system_prompt = """【Role & Task Definition】
You are a professional medical information processing specialist. Your task is to analyze medical dialogue texts and perform two types of annotation:

1. Identify and annotate semantically ambiguous words
2. Identify non-medical conversation fragments that should be removed

【Task 1: Ambiguity Detection】
Identify three types of ambiguity:
1. Medical Polysemy: Words with multiple specific meanings in a medical context.
Examples: mass (tumor, mass), lesion (pathological lesion, injury), block (heart block, blockage, nerve block).

2. Medical vs. Common Language Conflict: Words that have both an everyday meaning and a specific medical meaning.
Examples: cold (illness / low temperature), shock (medical shock / surprise), depression (clinical depression / economic downturn, geological hollow).

3. Ambiguous Medical Abbreviations: Abbreviations that can correspond to multiple full forms.
Examples: RA (Rheumatoid Arthritis / Right Atrium), MS (Multiple Sclerosis / Mitral Stenosis / Master of Science).

For ambiguous words, annotate using: [AMG:original_word]

【Task 2: Non-Medical Dialogue Detection】
Identify non-medical conversation fragments that are irrelevant to patient-doctor dialogue. Be very strict and comprehensive in detecting:

- Weather discussions (e.g., "yesterday weather very good", "it's sunny today")
- Small talk and casual conversation (e.g., "you know?", "like some computer problem")
- Environmental noise or unrelated topics
- Personal opinions not related to medical symptoms
- Technical issues unrelated to health (e.g., "computer problem")
- Any content that doesn't directly relate to medical symptoms, treatments, or health concerns

For non-medical fragments, annotate using: [NOS:start]...text...[NOS:end]

【Output Format & Rules】
1. For ambiguous words: replace the entire word with the tag [AMB:original_word].
2. For non-medical fragments: Only mark the parts of the sentence that appear to be unrelated to doctor–patient communication.
Wrap the entire fragment with [NOS:start] and [NOS:end]
3. Do NOT modify, correct, add, delete, or change any of the original words, spelling, or punctuation
4. Ensure you check every noun, adjective, and abbreviation for ambiguity
5. Be cautious: if there's any potential for confusion, you MUST annotate it

【Example】
Input: "After my cold, I've been feeling a lot of depression. The weather is nice today."
Output: "After my [AMB:cold], I've been feeling a lot of [AMB:depression]. [NOS:start]The weather is nice today.[NOS:end]" """
        
        user_prompt = f"Please analyze this medical dialogue text and perform both ambiguity detection and non-medical dialogue detection:\n\n{text}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                
                return response.choices[0].message.content.strip()
                
            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[CombinedMedicalDetector LLM 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[CombinedMedicalDetector LLM ERROR] {e}")
                    return text
            except Exception as e:
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[CombinedMedicalDetector LLM 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                        continue
                print(f"[CombinedMedicalDetector LLM ERROR] {e}")
                return text
        
        return text

    def _extract_ambiguity_tags(self, annotated_text: str) -> List[Tuple[str, int, int]]:
        """从标注文本中提取歧义标签信息"""
        ambiguity_tags = []
        pattern = r'\[AMBIG:([^\]]+)\]'
        
        for match in re.finditer(pattern, annotated_text):
            original_word = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            ambiguity_tags.append((original_word, start_pos, end_pos))
            
        return ambiguity_tags

    def _extract_non_medical_fragments(self, annotated_text: str) -> List[Tuple[str, int, int]]:
        """从标注文本中提取非医患对话片段"""
        non_medical_fragments = []
        pattern = r'\[NOS:start\](.*?)\[NOS:end\]'
        
        for match in re.finditer(pattern, annotated_text, re.DOTALL):
            fragment = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            non_medical_fragments.append((fragment, start_pos, end_pos))
            
        return non_medical_fragments

    def _protect_existing_tags(self, text: str) -> str:
        """保护已存在的标签，避免重复处理"""
        # 移除已存在的AMBIG和NOS标签，避免重复标注
        protected_text = re.sub(r'\[AMBIG:[^\]]+\]', '', text)
        protected_text = re.sub(r'\[NOS:start\].*?\[NOS:end\]', '', protected_text, flags=re.DOTALL)
        return protected_text

    def detect(self, text: str) -> List[SpanEdit]:
        """使用LLM同时检测歧义词和非医患对话"""
        if not self.client:
            return []
            
        # 保护措施：移除已存在的标签
        clean_text = self._protect_existing_tags(text)
        
        # 使用LLM同时检测歧义词和非医患对话
        annotated_text = self._detect_with_llm(clean_text)
        
        edits = []
        
        # 处理歧义标签
        ambiguity_tags = self._extract_ambiguity_tags(annotated_text)
        for original_word, tag_start, tag_end in ambiguity_tags:
            # 在原始文本中查找该词的位置
            word_pattern = re.escape(original_word)
            for match in re.finditer(word_pattern, text, re.IGNORECASE):
                word_start = match.start()
                word_end = match.end()
                
                # 检查该位置是否已经被其他编辑覆盖
                is_covered = False
                for existing_edit in edits:
                    if (word_start >= existing_edit.start_char and word_start < existing_edit.end_char) or \
                       (word_end > existing_edit.start_char and word_end <= existing_edit.end_char):
                        is_covered = True
                        break
                
                if not is_covered:
                    edits.append(SpanEdit(
                        start_char=word_start,
                        end_char=word_end,
                        op="REPLACE",
                        cand_texts=[f"[AMBIG:{original_word}]"],
                        score=0.8,  # 默认置信度
                        tag="AMB",
                        edit_type="candidate",
                        detector_name="CombinedMedical"
                    ))
                    break  # 只处理第一个匹配的词
        
        # 处理非医患对话片段
        non_medical_fragments = self._extract_non_medical_fragments(annotated_text)
        for fragment, tag_start, tag_end in non_medical_fragments:
            if fragment and fragment in text:
                start = text.find(fragment)
                if start != -1:
                    edits.append(SpanEdit(
                        start_char=start,
                        end_char=start + len(fragment),
                        op="DELETE",
                        cand_texts=[""],
                        score=0.9,
                        tag="NOS",
                        edit_type="deterministic",
                        detector_name="CombinedMedical"
                    ))
        
        return edits


# -------------------------- Editor Pipeline --------------------------
class EditManager:
    """Editor stage: Classify, score, and filter candidate edits"""
    
    def __init__(self, w1=0.3, w2=0.4, w3=0.3, delta=0.05):
        self.w1 = w1  # Edit cost weight
        self.w2 = w2  # Fluency weight  
        self.w3 = w3  # WSD confidence weight
        self.delta = delta  # Threshold for multi-candidate retention
        
        # 编辑优先级定义 (数值越小优先级越高)
        # 固定处理顺序：spell, repetition, grammar, ambiguity, nonmedical
        self.edit_priority = {
            "SPL": 1,  # 拼写错误 - 最高优先级（第1步）
            "RPT": 2,  # 重复检测 - 高优先级（第2步）
            "GRM": 3,  # 语法错误 - 中高优先级（第3步）
            "AMB": 4,  # 歧义消解 - 中优先级（第4步）
            "NOS": 5   # 非医学术语 - 低优先级（第5步）
        }
        
        # 编辑类型兼容性矩阵
        self.compatibility_matrix = {
            ("GRM", "SPL"): True,   # 语法和拼写可以合并
            ("GRM", "AMB"): False,  # 语法和歧义不兼容
            ("SPL", "AMB"): False,  # 拼写和歧义不兼容
            ("RPT", "GRM"): True,   # 重复和语法可以合并
            ("RPT", "SPL"): True,   # 重复和拼写可以合并
            ("NOS", "GRM"): False,  # 非医患对话与语法不兼容
            ("NOS", "SPL"): False,  # 非医患对话与拼写不兼容
        }
        
    def classify_edits(self, edits: List[SpanEdit]) -> Tuple[List[SpanEdit], List[SpanEdit]]:
        """Classify edits into deterministic vs candidate"""

        deterministic = []
        candidates = []
        
        for edit in edits:
            if edit.tag in ["RPT", "SPL", "GRM", "NOS", "AMB"]:
                edit.edit_type = "deterministic"
                deterministic.append(edit)
            else:
                # Default to deterministic for unknown tags
                edit.edit_type = "deterministic"
                deterministic.append(edit)
                
        return deterministic, candidates
    
    def calculate_edit_cost(self, original: str, candidate: str) -> float:
        """Calculate normalized edit cost (0-1, lower is better)"""
        if not original and not candidate:
            return 0.0
        if not original:
            return 1.0
        if not candidate:
            return 1.0
            
        # Levenshtein distance normalized by max length
        max_len = max(len(original), len(candidate))
        if max_len == 0:
            return 0.0
            
        # Simple character-level edit distance
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein(original, candidate)
        return distance / max_len
    
    def calculate_fluency_score(self, text: str) -> float:
        """Calculate pseudo-perplexity based fluency score"""
        # Simple heuristic: longer sentences with proper punctuation are more fluent
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
            
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        punctuation_score = len(re.findall(r'[.!?]', text)) / max(1, len(sentences))
        
        # Normalize to 0-1 range
        fluency = min(1.0, (avg_length / 10.0) * 0.5 + punctuation_score * 0.5)
        return fluency
    
    def calculate_wsd_confidence(self, edit: SpanEdit) -> float:
        """Calculate WSD confidence for ambiguous edits"""
        if edit.tag != "AMB":
            return 1.0
            
        # For AMB edits, use the score as confidence
        return edit.score
    
    def score_candidates(self, edit: SpanEdit, context: str) -> List[Tuple[str, float]]:
        """Score all candidates for an edit"""
        if not edit.cand_texts:
            return []
            
        original_text = context[edit.start_char:edit.end_char]
        scored_candidates = []
        
        for candidate in edit.cand_texts:
            # Calculate individual scores
            edit_cost = self.calculate_edit_cost(original_text, candidate)
            fluency = self.calculate_fluency_score(candidate)
            wsd_conf = self.calculate_wsd_confidence(edit)
            
            # Combined score
            score = (self.w1 * (1 - edit_cost) + 
                    self.w2 * fluency + 
                    self.w3 * wsd_conf)
            
            scored_candidates.append((candidate, score))
        
        return scored_candidates
    
    def filter_candidates(self, edit: SpanEdit, context: str) -> SpanEdit:
        """Filter and rank candidates based on scoring"""
        if edit.edit_type == "deterministic":
            return edit
            
        scored_candidates = self.score_candidates(edit, context)
        if not scored_candidates:
            return edit
            
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Check if top candidates are close in score
        if len(scored_candidates) > 1:
            top_score = scored_candidates[0][1]
            second_score = scored_candidates[1][1]
            
            if top_score - second_score < self.delta:
                # Keep multiple candidates for arbiter
                edit.cand_texts = [cand for cand, _ in scored_candidates[:3]]
            else:
                # Keep only top candidate
                edit.cand_texts = [scored_candidates[0][0]]
        else:
            edit.cand_texts = [scored_candidates[0][0]]
            
        return edit
    
    def merge_overlapping_edits(self, edits: List[SpanEdit]) -> List[SpanEdit]:
        if not edits:
            return []
            
        # 按优先级和位置排序
        sorted_edits = sorted(edits, key=lambda x: (self.edit_priority.get(x.tag, 999), x.start_char))
        merged = []
        
        for edit in sorted_edits:
            if not merged:
                merged.append(edit)
                continue
                
            last_edit = merged[-1]
            
            # 检查重叠
            if self._positions_overlap(edit, last_edit):
                # 智能合并策略
                merged_edit = self._smart_merge_edits(edit, last_edit)
                merged[-1] = merged_edit
            else:
                merged.append(edit)
                
        return merged
    
    def _smart_merge_edits(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """智能合并两个编辑"""
        # 检查兼容性
        if not self._check_compatibility(edit1, edit2):
            # 不兼容时，选择优先级更高的
            winner = self._get_priority_winner(edit1, edit2)
            return winner
        
        # 兼容时进行智能合并
        merged_candidates = self._merge_candidates(edit1, edit2)
        merged_score = self._calculate_merged_score(edit1, edit2)
        merged_tags = self._merge_tags(edit1.tag, edit2.tag)
        
        return SpanEdit(
            start_char=min(edit1.start_char, edit2.start_char),
            end_char=max(edit1.end_char, edit2.end_char),
            op=self._determine_merged_operation(edit1, edit2),
            cand_texts=merged_candidates,
            score=merged_score,
            tag=merged_tags,
            edit_type="candidate",
            detector_name=f"{edit1.detector_name}+{edit2.detector_name}"
        )
    
    def _merge_candidates(self, edit1: SpanEdit, edit2: SpanEdit) -> List[str]:
        """智能合并候选文本"""
        candidates1 = set(edit1.cand_texts)
        candidates2 = set(edit2.cand_texts)
        
        # 如果候选文本相同，直接返回
        if candidates1 == candidates2:
            return list(candidates1)
        
        # 合并候选文本，去重并保持顺序
        merged = []
        for cand in edit1.cand_texts:
            if cand not in merged:
                merged.append(cand)
        for cand in edit2.cand_texts:
            if cand not in merged:
                merged.append(cand)
        
        # 限制候选数量，避免过多
        return merged[:5]
    
    def _calculate_merged_score(self, edit1: SpanEdit, edit2: SpanEdit) -> float:
        """计算合并后的分数"""
        # 使用加权平均，权重基于优先级
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)
        
        # 优先级越高，权重越大
        weight1 = 1.0 / priority1
        weight2 = 1.0 / priority2
        total_weight = weight1 + weight2
        
        return (weight1 * edit1.score + weight2 * edit2.score) / total_weight
    
    def _merge_tags(self, tag1: str, tag2: str) -> str:
        """智能合并标签"""
        if tag1 == tag2:
            return tag1
        
        # 按优先级排序标签
        tags = [tag1, tag2]
        tags.sort(key=lambda x: self.edit_priority.get(x, 999))
        return "+".join(tags)
    
    def _determine_merged_operation(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """确定合并后的操作类型"""
        # 如果都是DELETE，保持DELETE
        if edit1.op == "DELETE" and edit2.op == "DELETE":
            return "DELETE"
        # 如果都是INSERT，保持INSERT
        elif edit1.op == "INSERT" and edit2.op == "INSERT":
            return "INSERT"
        # 其他情况使用REPLACE
        else:
            return "REPLACE"
    
    def process_edits(self, edits: List[SpanEdit], text: str) -> List[SpanEdit]:
        """Main processing pipeline for Editor stage"""
        # Classify edits (all are deterministic now)
        deterministic, candidates = self.classify_edits(edits)
        
        # 简化：不再进行评分和过滤，直接合并所有编辑
        all_processed = deterministic + candidates
        merged_edits = self.merge_overlapping_edits(all_processed)
        
        return merged_edits
    
    def _positions_overlap(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check if two edits have overlapping positions"""
        return (edit1.start_char < edit2.end_char and 
                edit2.start_char < edit1.end_char)
    
    def _check_compatibility(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """检查两个编辑是否兼容"""
        tag1, tag2 = edit1.tag, edit2.tag
        
        # 检查兼容性矩阵
        if (tag1, tag2) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag1, tag2)]
        if (tag2, tag1) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag2, tag1)]
        
        # 默认兼容性规则
        if tag1 == tag2:
            return True  # 相同类型兼容
        
        # 不同优先级类型通常不兼容
        return False
    
    def _get_priority_winner(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """根据优先级规则确定获胜编辑"""
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)
        
        if priority1 < priority2:
            return edit1
        elif priority2 < priority1:
            return edit2
        else:
            # 优先级相同时，选择分数更高的
            return edit1 if edit1.score >= edit2.score else edit2
    
    def _merge_tags(self, tag1: str, tag2: str) -> str:
        """合并标签"""
        if tag1 == tag2:
            return tag1
        
        # 根据优先级选择主要标签
        priority1 = self.edit_priority.get(tag1, 999)
        priority2 = self.edit_priority.get(tag2, 999)
        
        if priority1 <= priority2:
            return tag1
        else:
            return tag2
    
    def _determine_merged_operation(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """确定合并后的操作类型"""
        # 如果两个编辑都是DELETE，合并后仍为DELETE
        if edit1.op == "DELETE" and edit2.op == "DELETE":
            return "DELETE"
        
        # 如果两个编辑都是INSERT，合并后仍为INSERT
        if edit1.op == "INSERT" and edit2.op == "INSERT":
            return "INSERT"
        
        # 其他情况默认为REPLACE
        return "REPLACE"

class EditorPipeline:
    """Editor stage pipeline"""
    
    def __init__(self, w1=0.3, w2=0.4, w3=0.3, delta=0.05, api_key: str = None):
        self.edit_manager = EditManager(w1, w2, w3, delta)
        self.api_key = api_key
        
    def _interpret_ambiguity_word(self, word: str, context: str) -> str:
        """为歧义词生成英文医学注释，返回word(interpretation)格式"""
        if not self.api_key:
            return word
            
        from openai import OpenAI
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.chatanywhere.tech/v1",
            timeout=30.0  # 设置30秒超时
        )
        
        system_prompt = """You are a medical text interpretation specialist. Your task is to provide a concise English medical interpretation for ambiguous words in medical contexts.

Rules:
1. Provide a simple, professional English interpretation (2-5 words)
2. Consider the medical context when interpreting
3. Return ONLY the interpretation text, nothing else
4. The interpretation should clarify the medical meaning of the word

Example:
Word: "mass" in context "The patient has a mass in the lung."
Interpretation: "abnormal tissue growth"

Example:
Word: "depression" in context "I have depression."
Interpretation: "clinical mood disorder"

Example:
Word: "cold" in context "After my cold, I've been feeling tired."
Interpretation: "common illness" """
        
        user_prompt = f"Word: \"{word}\"\nContext: \"{context}\"\n\nPlease provide a concise English medical interpretation for this word in this context. Return ONLY the interpretation (2-5 words), nothing else."
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                
                interpretation = response.choices[0].message.content.strip()
                
                # 清理结果，移除可能的引号或解释性文本
                if interpretation.startswith('"') and interpretation.endswith('"'):
                    interpretation = interpretation[1:-1]
                if interpretation.startswith("'") and interpretation.endswith("'"):
                    interpretation = interpretation[1:-1]
                if "interpretation:" in interpretation.lower():
                    interpretation = interpretation.split(":")[-1].strip()
                
                # 返回word(interpretation)格式
                return f"{word}({interpretation})" if interpretation else word
                
            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[EditorPipeline Interpretation 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[EditorPipeline Interpretation ERROR] {e}")
                    return word
            except Exception as e:
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[EditorPipeline Interpretation 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                        continue
                print(f"[EditorPipeline Interpretation ERROR] {e}")
                return word
        
        return word
    
    def run(self, edits: List[SpanEdit], text: str) -> Dict:
        """Run editor pipeline"""
        # 处理编辑：合并重叠编辑
        processed_edits = self.edit_manager.process_edits(edits, text)
        
        # 处理AMB编辑：将[AMBIG:word]转换为word(interpretation)格式
        for edit in processed_edits:
            if edit.tag == "AMB" and edit.cand_texts:
                # 提取原始词（从[AMBIG:word]格式中提取）
                import re
                amb_tag = edit.cand_texts[0] if edit.cand_texts else ""
                match = re.match(r'\[AMBIG:([^\]]+)\]', amb_tag)
                if match:
                    original_word = match.group(1)
                    # 获取上下文（包含该词的句子）
                    sentence_start = max(0, text.rfind('.', 0, edit.start_char) + 1, 
                                        text.rfind('!', 0, edit.start_char) + 1,
                                        text.rfind('?', 0, edit.start_char) + 1)
                    sentence_end = len(text)
                    for punct in ['.', '!', '?']:
                        end_pos = text.find(punct, edit.end_char)
                        if end_pos != -1:
                            sentence_end = min(sentence_end, end_pos + 1)
                    context = text[sentence_start:sentence_end].strip()
                    
                    # 生成注释格式
                    annotated_word = self._interpret_ambiguity_word(original_word, context)
                    edit.cand_texts = [annotated_word]
        
        # 所有编辑都是deterministic，直接应用
        edited_text = self.apply_deterministic_edits(text, processed_edits)
        
        return {
            "processed_edits": processed_edits,
            "deterministic_edits": processed_edits,  # 所有编辑都是deterministic
            "candidate_edits": [],  # 不再有候选编辑
            "edited_text": edited_text,
            "interpreted_text": text  # 不再单独处理interpreted_text
        }
    
    def apply_deterministic_edits(self, text: str, edits: List[SpanEdit]) -> str:
        """Apply deterministic edits to text"""
        if not edits:
            return text
            
        # Sort edits by position (reverse order to maintain indices)
        sorted_edits = sorted(edits, key=lambda x: x.start_char, reverse=True)
        
        result = text
        for edit in sorted_edits:
            start = edit.start_char
            end = edit.end_char
            
            if edit.op == "DELETE":
                result = result[:start] + result[end:]
            elif edit.op == "REPLACE":
                replacement = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + replacement + result[end:]
            elif edit.op == "INSERT":
                insertion = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + insertion + result[start:]
                
        return result

# -------------------------- Arbiter Pipeline --------------------------
class ArbiterCore:
    """Arbiter core: Conflict detection and candidate evaluation"""
    
    def __init__(self):
        self.conflict_threshold = 0.1  # Threshold for score differences
        
        # 编辑优先级规则 (数字越小优先级越高)
        # 固定处理顺序：spell, repetition, grammar, ambiguity, nonmedical
        self.edit_priority = {
            "SPL": 1,  # 拼写错误 - 最高优先级（第1步）
            "RPT": 2,  # 重复检测 - 高优先级（第2步）
            "GRM": 3,  # 语法错误 - 中高优先级（第3步）
            "AMB": 4,  # 歧义消解 - 中优先级（第4步）
            "NOS": 5,  # 非医患对话 - 低优先级（第5步）
            "Fragment": 6  # 片段噪声 - 最低优先级（已移除，保留以防兼容性）
        }
        
        # 编辑类型兼容性矩阵
        self.compatibility_matrix = {
            ("GRM", "SPL"): True,   # 语法和拼写可以合并
            ("GRM", "AMB"): False,  # 语法和歧义不兼容
            ("SPL", "AMB"): False,  # 拼写和歧义不兼容
            ("RPT", "GRM"): True,   # 重复和语法可以合并
            ("RPT", "SPL"): True,   # 重复和拼写可以合并
            ("NOS", "GRM"): False,  # 非医患对话与语法不兼容
            ("NOS", "SPL"): False,  # 非医患对话与拼写不兼容
        }
        
    def detect_conflicts(self, edits: List[SpanEdit]) -> List[Dict]:
        """Detect position and candidate conflicts with enhanced analysis"""
        conflicts = []
        
        # Check for position conflicts with priority analysis
        for i, edit1 in enumerate(edits):
            for j, edit2 in enumerate(edits[i+1:], i+1):
                if self._positions_overlap(edit1, edit2):
                    # 分析冲突类型和优先级
                    conflict_type = self._analyze_position_conflict(edit1, edit2)
                    conflicts.append({
                        "type": "position_conflict",
                        "edit1": edit1,
                        "edit2": edit2,
                        "overlap_span": (max(edit1.start_char, edit2.start_char),
                                       min(edit1.end_char, edit2.end_char)),
                        "conflict_type": conflict_type,
                        "priority_winner": self._get_priority_winner(edit1, edit2),
                        "compatibility": self._check_compatibility(edit1, edit2)
                    })
        
        # Check for candidate conflicts
        for edit in edits:
            if len(edit.cand_texts) > 1:
                conflicts.append({
                    "type": "candidate_conflict", 
                    "edit": edit,
                    "candidates": edit.cand_texts,
                    "candidate_scores": self._score_all_candidates(edit)
                })
                
        return conflicts
    
    def _positions_overlap(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """Check if two edits have overlapping positions"""
        return (edit1.start_char < edit2.end_char and 
                edit2.start_char < edit1.end_char)
    
    def _analyze_position_conflict(self, edit1: SpanEdit, edit2: SpanEdit) -> str:
        """分析位置冲突的类型"""
        # 检查是否完全重叠
        if (edit1.start_char == edit2.start_char and edit1.end_char == edit2.end_char):
            return "exact_overlap"
        # 检查是否包含关系
        elif (edit1.start_char <= edit2.start_char and edit1.end_char >= edit2.end_char):
            return "edit1_contains_edit2"
        elif (edit2.start_char <= edit1.start_char and edit2.end_char >= edit1.end_char):
            return "edit2_contains_edit1"
        # 检查是否部分重叠
        else:
            return "partial_overlap"
    
    def _get_priority_winner(self, edit1: SpanEdit, edit2: SpanEdit) -> SpanEdit:
        """根据优先级规则确定获胜编辑"""
        priority1 = self.edit_priority.get(edit1.tag, 999)
        priority2 = self.edit_priority.get(edit2.tag, 999)
        
        if priority1 < priority2:
            return edit1
        elif priority2 < priority1:
            return edit2
        else:
            # 优先级相同时，选择分数更高的
            return edit1 if edit1.score >= edit2.score else edit2
    
    def _check_compatibility(self, edit1: SpanEdit, edit2: SpanEdit) -> bool:
        """检查两个编辑是否兼容"""
        tag1, tag2 = edit1.tag, edit2.tag
        
        # 检查直接兼容性
        if (tag1, tag2) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag1, tag2)]
        elif (tag2, tag1) in self.compatibility_matrix:
            return self.compatibility_matrix[(tag2, tag1)]
        
        # 默认兼容性规则
        return tag1 == tag2  # 相同标签默认兼容
    
    def _score_all_candidates(self, edit: SpanEdit) -> Dict[str, float]:
        """为编辑的所有候选评分"""
        scores = {}
        for candidate in edit.cand_texts:
            # 使用简化的评分方法
            score = self._calculate_comprehensive_score(edit, candidate, "")
            scores[candidate] = score
        return scores
    
    def evaluate_candidates(self, edit: SpanEdit, context: str) -> Dict:
        """Evaluate all candidates for an edit"""
        if not edit.cand_texts:
            return {"best_candidate": "", "scores": {}}
            
        scores = {}
        for candidate in edit.cand_texts:
            score = self._calculate_comprehensive_score(edit, candidate, context)
            scores[candidate] = score
            
        # Find best candidate
        best_candidate = max(scores.keys(), key=lambda x: scores[x])
        
        return {
            "best_candidate": best_candidate,
            "scores": scores,
            "confidence": scores[best_candidate]
        }
    
    def _calculate_comprehensive_score(self, edit: SpanEdit, candidate: str, context: str) -> float:
        """Calculate comprehensive score for a candidate"""
        # Edit cost (lower is better)
        original_text = context[edit.start_char:edit.end_char]
        edit_cost = self._calculate_edit_cost(original_text, candidate)
        
        # Fluency score
        fluency = self._calculate_fluency(candidate)
        
        # Semantic consistency (simplified)
        semantic_consistency = self._calculate_semantic_consistency(original_text, candidate, context)
        
        # Term preservation (for medical terms)
        term_preservation = self._calculate_term_preservation(original_text, candidate)
        
        # Repetition penalty
        repetition_penalty = self._calculate_repetition_penalty(candidate, context)
        
        # Weighted combination
        score = (0.2 * (1 - edit_cost) + 
                0.3 * fluency + 
                0.2 * semantic_consistency + 
                0.2 * term_preservation + 
                0.1 * (1 - repetition_penalty))
        
        return max(0.0, min(1.0, score))
    
    def _calculate_edit_cost(self, original: str, candidate: str) -> float:
        """Calculate normalized edit cost"""
        if not original and not candidate:
            return 0.0
        max_len = max(len(original), len(candidate))
        if max_len == 0:
            return 0.0
            
        # Simple character-level distance
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        distance = levenshtein(original, candidate)
        return distance / max_len
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score"""
        if not text:
            return 0.0
            
        # Simple heuristics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]', text)) + 1
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Punctuation usage
        punctuation_score = len(re.findall(r'[.!?,;:]', text)) / max(1, word_count)
        
        # Combine factors
        fluency = min(1.0, (avg_sentence_length / 15.0) * 0.7 + punctuation_score * 10 * 0.3)
        return fluency
    
    def _calculate_semantic_consistency(self, original: str, candidate: str, context: str) -> float:
        """Calculate semantic consistency score"""
        # Simplified: check if candidate maintains similar word patterns
        original_words = set(original.lower().split())
        candidate_words = set(candidate.lower().split())
        
        if not original_words and not candidate_words:
            return 1.0
        if not original_words or not candidate_words:
            return 0.5
            
        # Jaccard similarity
        intersection = len(original_words & candidate_words)
        union = len(original_words | candidate_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_term_preservation(self, original: str, candidate: str) -> float:
        """Calculate medical term preservation score"""
        # Simple check for medical-like terms (capitalized words, common medical suffixes)
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:itis|osis|emia|uria|algia|pathy)\b',  # Medical suffixes
            r'\b(?:heart|brain|liver|lung|kidney|blood|pain|fever|temperature)\b',  # Common medical terms
        ]
        
        original_medical = sum(len(re.findall(pattern, original, re.IGNORECASE)) for pattern in medical_patterns)
        candidate_medical = sum(len(re.findall(pattern, candidate, re.IGNORECASE)) for pattern in medical_patterns)
        
        if original_medical == 0:
            return 1.0 if candidate_medical == 0 else 0.8
            
        preservation_ratio = candidate_medical / original_medical
        return min(1.0, preservation_ratio)
    
    def _calculate_repetition_penalty(self, candidate: str, context: str) -> float:
        """Calculate repetition penalty"""
        if not candidate:
            return 0.0
            
        # Check for repeated words in candidate
        words = candidate.split()
        if len(words) <= 1:
            return 0.0
            
        # Count repeated words
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        # Normalize by total words
        repetition_ratio = repeated_words / len(words)
        return min(1.0, repetition_ratio)

class ArbiterPipeline:
    """Enhanced Arbiter stage pipeline with intelligent conflict resolution"""
    
    def __init__(self, api_key: str = None):
        self.arbiter_core = ArbiterCore()
        self.api_key = api_key
    
    def run(self, edits: List[SpanEdit], original_text: str, editor_processed_text: str) -> Dict:
        """
        Run enhanced arbiter pipeline with intelligent conflict resolution
        
        Args:
            edits: 编辑列表
            original_text: 输入整个系统的原始句子
            editor_processed_text: Editor处理之后的句子
        """
        # Detect conflicts with enhanced analysis
        conflicts = self.arbiter_core.detect_conflicts(edits)
        
        # Resolve conflicts using intelligent strategies
        resolved_edits = self._resolve_conflicts_intelligently(edits, editor_processed_text, conflicts)
        
        # Apply all resolved edits
        edited_text = self.apply_resolved_edits(editor_processed_text, resolved_edits)
        
        # Use LLM to check and correct the editor-processed sentence
        final_text = self._check_and_correct_editor_output(original_text, editor_processed_text, edited_text)
        
        return {
            "conflicts": conflicts,
            "resolved_edits": resolved_edits,
            "final_text": final_text,
            "edited_text": edited_text,
            "resolution_strategy": "intelligent_priority_based"
        }
    
    def _resolve_conflicts_intelligently(self, edits: List[SpanEdit], text: str, conflicts: List[Dict]) -> List[SpanEdit]:
        """智能冲突解决策略"""
        # 创建编辑映射，用于跟踪处理状态
        edit_map = {id(edit): edit for edit in edits}
        processed_edits = set()
        resolved_edits = []
        
        # 按优先级排序编辑
        sorted_edits = sorted(edits, key=lambda x: (
            self.arbiter_core.edit_priority.get(x.tag, 999),
            -x.score,  # 分数高的优先
            x.start_char  # 位置靠前的优先
        ))
        
        for edit in sorted_edits:
            if id(edit) in processed_edits:
                continue
                
            # 检查是否有位置冲突
            conflicting_edits = self._find_conflicting_edits(edit, edit_map, processed_edits)
            
            if conflicting_edits:
                # 处理冲突
                resolved_edit = self._resolve_position_conflicts(edit, conflicting_edits, text)
                resolved_edits.append(resolved_edit)
                
                # 标记所有冲突编辑为已处理
                processed_edits.add(id(edit))
                for conflict_edit in conflicting_edits:
                    processed_edits.add(id(conflict_edit))
            else:
                # 无冲突，直接处理
                resolved_edit = self._resolve_single_edit(edit, text)
                resolved_edits.append(resolved_edit)
                processed_edits.add(id(edit))
        
        return resolved_edits
    
    def _find_conflicting_edits(self, edit: SpanEdit, edit_map: Dict, processed_edits: set) -> List[SpanEdit]:
        """查找与给定编辑冲突的编辑"""
        conflicting = []
        for other_edit in edit_map.values():
            if (id(other_edit) != id(edit) and 
                id(other_edit) not in processed_edits and
                self.arbiter_core._positions_overlap(edit, other_edit)):
                conflicting.append(other_edit)
        return conflicting
    
    def _resolve_position_conflicts(self, primary_edit: SpanEdit, conflicting_edits: List[SpanEdit], text: str) -> SpanEdit:
        """解决位置冲突"""
        # 检查兼容性
        compatible_edits = [primary_edit]
        for conflict_edit in conflicting_edits:
            if self.arbiter_core._check_compatibility(primary_edit, conflict_edit):
                compatible_edits.append(conflict_edit)
        
        if len(compatible_edits) > 1:
            # 兼容编辑可以合并
            return self._merge_compatible_edits(compatible_edits, text)
        else:
            # 不兼容，选择优先级最高的
            all_edits = [primary_edit] + conflicting_edits
            winner = self.arbiter_core._get_priority_winner(primary_edit, conflicting_edits[0])
            return self._resolve_single_edit(winner, text)
    
    def _merge_compatible_edits(self, compatible_edits: List[SpanEdit], text: str) -> SpanEdit:
        """合并兼容的编辑"""
        if len(compatible_edits) == 1:
            return self._resolve_single_edit(compatible_edits[0], text)
        
        # 合并所有兼容编辑
        merged_candidates = []
        merged_score = 0
        merged_tags = []
        
        for edit in compatible_edits:
            merged_candidates.extend(edit.cand_texts)
            merged_score += edit.score
            merged_tags.append(edit.tag)
        
        # 去重候选文本
        merged_candidates = list(set(merged_candidates))
        
        # 计算平均分数
        merged_score = merged_score / len(compatible_edits)
        
        # 合并标签
        merged_tags = "+".join(sorted(set(merged_tags)))
        
        # 创建合并后的编辑
        merged_edit = SpanEdit(
            start_char=min(e.start_char for e in compatible_edits),
            end_char=max(e.end_char for e in compatible_edits),
            op="REPLACE",
            cand_texts=merged_candidates,
            score=merged_score,
            tag=merged_tags,
            edit_type="candidate",
            detector_name="+".join(set(e.detector_name for e in compatible_edits))
        )
        
        return self._resolve_single_edit(merged_edit, text)
    
    def _resolve_single_edit(self, edit: SpanEdit, text: str) -> SpanEdit:
        """解决单个编辑"""
        if edit.edit_type == "deterministic":
            return edit
        
        # 评估候选
        evaluation = self.arbiter_core.evaluate_candidates(edit, text)
        
        # 使用评估结果选择最佳候选（LLM澄清现在在_generate_fluent_sentence中处理）
        best_candidate = evaluation["best_candidate"]
        
        # 创建最终编辑
        return SpanEdit(
            start_char=edit.start_char,
            end_char=edit.end_char,
            op=edit.op,
            cand_texts=[best_candidate],
            score=evaluation["confidence"],
            tag=edit.tag,
            edit_type="deterministic",
            detector_name=edit.detector_name
        )
    
    def apply_resolved_edits(self, text: str, edits: List[SpanEdit]) -> str:
        """Apply all resolved edits to produce final text"""
        if not edits:
            return text
            
        # Sort edits by position (reverse order to maintain indices)
        sorted_edits = sorted(edits, key=lambda x: x.start_char, reverse=True)
        
        result = text
        for edit in sorted_edits:
            start = edit.start_char
            end = edit.end_char
            
            if edit.op == "DELETE":
                result = result[:start] + result[end:]
            elif edit.op == "REPLACE":
                replacement = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + replacement + result[end:]
            elif edit.op == "INSERT":
                insertion = edit.cand_texts[0] if edit.cand_texts else ""
                result = result[:start] + insertion + result[start:]
                
        return result
    
    def _check_and_correct_editor_output(self, original_text: str, editor_processed_text: str, edited_text: str) -> str:
        """
        检查editor处理后的句子和原句子相比是否处理正确，并进行修正
        
        Args:
            original_text: 输入整个系统的原始句子
            editor_processed_text: Editor处理之后的句子
            edited_text: 应用了arbiter resolved edits后的文本
            
        Returns:
            str: 修正后的最终文本
        """
        if not self.api_key:
            return edited_text
        
        from openai import OpenAI
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.chatanywhere.tech/v1",
            timeout=30.0  # 设置30秒超时
        )
        
        system_prompt = """You are a medical text quality checker. Your task is to review the editor-processed sentence and compare it with the original sentence to ensure correct processing.

You need to check:
1. Whether non-medical dialogue fragments are properly deleted (no missed deletions)
2. Whether useful medical information is accidentally deleted (no over-deletion)
3. Whether AMB word interpretations are accurate
4. Whether the sentence is fluent and natural
5. Whether the processed sentence contains the complete meaning of the original question without missing any key information
6. Whether the generated sentence contains any tags (should be removed)

If any issues are found, provide a corrected version. Return ONLY the corrected sentence, no explanations."""
        
        user_prompt = f"""Original sentence (input to the system): {original_text}

Editor-processed sentence: {editor_processed_text}

Current processed text (after arbiter edits): {edited_text}

Please check if the editor-processed sentence is correct compared to the original sentence. Check:
1. Are all non-medical dialogue fragments deleted? (no missed deletions)
2. Is any useful medical information accidentally deleted? (no over-deletion)
3. Are AMB word interpretations accurate?
4. Is the sentence fluent and natural?
5. Does it contain the complete meaning of the original question without missing key information?
6. Are there any tags remaining? (should be removed)

If corrections are needed, provide the corrected sentence. Return ONLY the final corrected sentence, no explanations."""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                result = response.choices[0].message.content.strip()
                
                # 清理结果，移除可能的解释性文本
                if "Corrected sentence:" in result:
                    result = result.split("Corrected sentence:")[-1].strip()
                if "Final sentence:" in result:
                    result = result.split("Final sentence:")[-1].strip()
                if "The corrected sentence is:" in result:
                    result = result.split("The corrected sentence is:")[-1].strip()
                
                # 移除引号
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                if result.startswith("'") and result.endswith("'"):
                    result = result[1:-1]
                
                # 确保没有残留的标签
                import re
                result = re.sub(r'\[AMB:[^\]]+\]', '', result)
                result = re.sub(r'\[AMBIG:[^\]]+\]', '', result)
                result = re.sub(r'\[NOS:start\].*?\[NOS:end\]', '', result, flags=re.DOTALL)
                
                return result if result else edited_text
                
            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[ArbiterPipeline Check and Correct 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ArbiterPipeline Check and Correct ERROR] {e}")
                    return edited_text
            except Exception as e:
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[ArbiterPipeline Check and Correct 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                        continue
                print(f"[ArbiterPipeline Check and Correct ERROR] {e}")
                return edited_text
        
        return edited_text

# -------------------------- Evaluation Metrics --------------------------
class MedicalTermsManager:
    """医学词典管理器 - 高效检索医学术语"""
    
    def __init__(self, dictionary_path: str = None):
        self.dictionary_path = dictionary_path
        self._terms_set = None
        self._terms_trie = None
        self._regex_patterns_cache = {}
        self._loaded = False
        
    def load_medical_dictionary(self, dictionary_path: str = None) -> bool:
        """加载医学词典文件"""
        if dictionary_path:
            self.dictionary_path = dictionary_path
            
        if not self.dictionary_path:
            print("未指定医学词典文件路径")
            return False
            
        try:
            import json
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取术语
            terms = set()
            for item in data:
                if isinstance(item, dict) and "term" in item:
                    terms.add(item["term"])
            
            self._terms_set = terms
            self._build_trie()
            self._loaded = True
            
            print(f"成功加载 {len(terms)} 个医学术语")
            return True
            
        except Exception as e:
            print(f"加载医学词典失败: {e}")
            return False
    
    def _build_trie(self):
        """构建Trie树用于快速前缀匹配"""
        if not self._terms_set:
            return
            
        self._terms_trie = {}
        for term in self._terms_set:
            node = self._terms_trie
            for char in term.lower():
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = True  # 标记单词结束
    
    def is_medical_term(self, token: str) -> bool:
        """快速检查是否为医学术语"""
        if not self._loaded:
            return False
        return token.lower() in self._terms_set
    
    def get_medical_terms(self) -> set:
        """获取医学术语集合"""
        return self._terms_set or set()
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        if not self._loaded:
            return {"status": "not_loaded"}
            
        term_count = len(self._terms_set)
        return {
            "total_terms": term_count,
            "recommended_method": self._get_recommended_method(term_count),
            "estimated_time": self._estimate_processing_time(term_count),
            "memory_usage": f"{term_count * 20 / 1024 / 1024:.2f} MB"
        }
    
    def _get_recommended_method(self, term_count: int) -> str:
        """根据术语数量推荐最佳方法"""
        if term_count > 10000:
            return "正则表达式批量匹配（推荐）"
        elif term_count > 1000:
            return "正则表达式批量匹配"
        elif term_count > 100:
            return "优化的子字符串匹配"
        else:
            return "简单计数方法"
    
    def _estimate_processing_time(self, term_count: int) -> str:
        """估算处理时间"""
        if term_count > 100000:
            return "> 10秒（大词典）"
        elif term_count > 10000:
            return "1-5秒（中等词典）"
        elif term_count > 1000:
            return "0.1-1秒（小词典）"
        else:
            return "< 0.1秒（微型词典）"

class DenoisingQualityGEval:
    """GEval评分器，用于去噪质量评价（Accuracy, Integrity, Smoothness）"""
    
    def __init__(self, api_key: str = None, model_name: str = "deepseek-v3", base_url: str = None):
        """
        初始化去噪质量评分器
        
        Args:
            api_key: OpenAI API密钥
            model_name: 使用的模型名称
            base_url: API基础URL
        """
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.chatanywhere.tech/v1",
                timeout=60.0  # 设置60秒超时
            )
            self.model_name = model_name
        else:
            self.client = None
            self.model_name = model_name
    
    def evaluate(self, original_text: str, denoised_text: str) -> Dict[str, float]:
        """
        对去噪后的文本进行多维度评分
        
        Args:
            original_text: 原始文本（未经过detector和editor处理的原始输入）
            denoised_text: 去噪后的文本
            
        Returns:
            Dict: 包含Accuracy, Integrity, Smoothness三个维度评分的字典
        """
        if not self.client:
            return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        
        prompt = f"""请对以下去噪后的医疗对话文本进行评分，每个维度1-5分：

原始文本（未处理的输入）: {original_text}

去噪后的文本: {denoised_text}

评分维度：
1. 准确性 (Accuracy): 评估去噪后文本的医学信息准确性，是否保留了原始文本的核心医学含义，是否纠正了错误但未引入新的错误
2. 完整性 (Integrity): 评估去噪后文本是否完整保留了原始文本的重要医学信息，是否丢失了关键内容
3. 流畅性 (Smoothness): 评估去噪后文本的语言表达是否流畅自然，是否符合医疗对话的表达习惯

请以JSON格式输出评分结果，格式如下：
{{"accuracy": 分数, "integrity": 分数, "smoothness": 分数}}

只输出JSON格式的评分结果，不要包含其他文字。"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个专业的医疗文本质量评估专家。请客观公正地对去噪后的医疗对话文本进行评分，关注医学信息的准确性、完整性和语言流畅性。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                
                # 解析JSON响应
                rating_text = response.choices[0].message.content.strip()
                # 尝试提取JSON部分
                json_match = re.search(r'\{.*\}', rating_text, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    # 确保所有维度都有评分
                    result = {}
                    for dim in ["accuracy", "integrity", "smoothness"]:
                        if dim in scores:
                            result[dim] = float(scores[dim])
                        else:
                            result[dim] = 4.0  # 默认分数
                    return result
                else:
                    # 如果无法解析JSON，返回默认评分
                    return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
                    
            except (APIConnectionError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[DenoisingQualityGEval 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[DenoisingQualityGEval ERROR] {e}")
                    return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
            except Exception as e:
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[DenoisingQualityGEval 重试 {attempt+1}/{max_retries}] 等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                        continue
                print(f"[DenoisingQualityGEval ERROR] {e}")
                return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        
        return {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}

class EvaluationMetrics:
    """Evaluation metrics for the DEA framework"""
    
    def __init__(self, medical_terms_manager=None):
        self.medical_terms_manager = medical_terms_manager
        self._regex_patterns_cache = {}  # 缓存正则表达式模式
    
    def calculate_consistency(self, original: str, denoised: str) -> float:
        """Calculate semantic consistency between original and denoised text"""
        # Simple word overlap-based consistency
        original_words = set(original.lower().split())
        denoised_words = set(denoised.lower().split())
        
        if not original_words and not denoised_words:
            return 1.0
        if not original_words or not denoised_words:
            return 0.0
            
        intersection = len(original_words & denoised_words)
        union = len(original_words | denoised_words)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_medical_accuracy(self, text: str, medical_terms: set = None) -> float:
        """Calculate medical accuracy based on preserved medical terms"""
        # 优先使用医学词典管理器
        if self.medical_terms_manager and self.medical_terms_manager._loaded:
            medical_terms = self.medical_terms_manager.get_medical_terms()
        elif not medical_terms:
            return 1.0
            
        if not medical_terms:
            return 1.0
            
        # 使用优化的术语匹配
        preserved_terms = self._count_preserved_medical_terms(text, medical_terms)
        
        return preserved_terms / len(medical_terms) if medical_terms else 1.0
    
    def calculate_medical_term_retention_rate(self, original_text: str, denoised_text: str) -> float:
        """
        计算医学术语保留率
        
        Args:
            original_text: 原始输入文本
            denoised_text: Arbiter最终输出文本
            
        Returns:
            float: 术语保留率 = arbiter输出句子的医学词汇数量 / 原始文本的医学词汇数量
        """
        # 获取医学术语集合
        if not self.medical_terms_manager or not self.medical_terms_manager._loaded:
            return 1.0  # 如果没有词典，返回1.0
        
        medical_terms = self.medical_terms_manager.get_medical_terms()
        if not medical_terms:
            return 1.0
        
        # 统计原始文本中的医学术语数量
        original_term_count = self._count_preserved_medical_terms(original_text, medical_terms)
        
        # 统计去噪后文本中的医学术语数量
        denoised_term_count = self._count_preserved_medical_terms(denoised_text, medical_terms)
        
        # 计算保留率
        if original_term_count == 0:
            return 1.0  # 如果原始文本没有医学术语，返回1.0
        
        retention_rate = denoised_term_count / original_term_count
        return retention_rate
    
    def _count_preserved_medical_terms(self, text: str, medical_terms: set) -> int:
        """优化的医学术语计数方法"""
        if not medical_terms:
            return 0
            
        text_lower = text.lower()
        preserved_count = 0
        
        # 方案1：使用正则表达式批量匹配（推荐用于大词典）
        if len(medical_terms) > 1000:
            return self._regex_based_count(text_lower, medical_terms)
        
        # 方案2：优化的子字符串匹配（适用于中等规模词典）
        elif len(medical_terms) > 100:
            return self._optimized_substring_count(text_lower, medical_terms)
        
        # 方案3：原始方法（适用于小词典）
        else:
            return self._simple_count(text_lower, medical_terms)
    
    def _regex_based_count(self, text_lower: str, medical_terms: set) -> int:
        """基于正则表达式的高效匹配（适用于大词典）"""
        import re
        
        # 使用缓存避免重复构建正则表达式
        terms_key = frozenset(medical_terms)
        if terms_key not in self._regex_patterns_cache:
            # 构建正则表达式模式
            # 转义特殊字符并创建单词边界匹配
            escaped_terms = [re.escape(term.lower()) for term in medical_terms]
            pattern = r'\b(?:' + '|'.join(escaped_terms) + r')\b'
            self._regex_patterns_cache[terms_key] = re.compile(pattern, re.IGNORECASE)
        
        compiled_pattern = self._regex_patterns_cache[terms_key]
        
        # 一次性匹配所有术语
        matches = compiled_pattern.findall(text_lower)
        
        # 统计唯一匹配的术语数量
        unique_matches = set(matches)
        return len(unique_matches)
    
    def _optimized_substring_count(self, text_lower: str, medical_terms: set) -> int:
        """优化的子字符串匹配（适用于中等规模词典）"""
        preserved_count = 0
        
        # 按长度排序，优先匹配长术语（避免短术语误匹配）
        sorted_terms = sorted(medical_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            if term.lower() in text_lower:
                preserved_count += 1
                # 可选：从文本中移除已匹配的术语，避免重复匹配
                # text_lower = text_lower.replace(term.lower(), '', 1)
        
        return preserved_count
    
    def _simple_count(self, text_lower: str, medical_terms: set) -> int:
        """简单计数方法（适用于小词典）"""
        return sum(1 for term in medical_terms if term.lower() in text_lower)
    
    def load_large_medical_dictionary(self, file_path: str, format: str = "txt") -> set:
        """
        加载大型医疗词典文件
        
        Args:
            file_path: 词典文件路径
            format: 文件格式 ("txt", "csv", "json")
            
        Returns:
            set: 医学术语集合
        """
        medical_terms = set()
        
        try:
            if format == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            medical_terms.add(term)
            
            elif format == "csv":
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0].strip():
                            medical_terms.add(row[0].strip())
            
            elif format == "json":
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        medical_terms = set(data)
                    elif isinstance(data, dict) and "terms" in data:
                        medical_terms = set(data["terms"])
            
            print(f"✅ 成功加载 {len(medical_terms)} 个医学术语")
            return medical_terms
            
        except Exception as e:
            print(f"❌ 加载医疗词典失败: {e}")
            return set()
    
    def get_performance_stats(self, medical_terms: set) -> dict:
        """获取性能统计信息"""
        return {
            "total_terms": len(medical_terms),
            "recommended_method": self._get_recommended_method(len(medical_terms)),
            "estimated_time": self._estimate_processing_time(len(medical_terms))
        }
    
    def _get_recommended_method(self, term_count: int) -> str:
        """根据术语数量推荐最佳方法"""
        if term_count > 10000:
            return "正则表达式批量匹配（推荐）"
        elif term_count > 1000:
            return "正则表达式批量匹配"
        elif term_count > 100:
            return "优化的子字符串匹配"
        else:
            return "简单计数方法"
    
    def _estimate_processing_time(self, term_count: int) -> str:
        """估算处理时间"""
        if term_count > 100000:
            return "> 10秒（大词典）"
        elif term_count > 10000:
            return "1-5秒（中等词典）"
        elif term_count > 1000:
            return "0.1-1秒（小词典）"
        else:
            return "< 0.1秒（微型词典）"
    
    
    def calculate_correctness(self, denoised: str, gold_standard: str) -> float:
        """Calculate correctness against gold standard"""
        if not gold_standard:
            return 0.0
            
        # Simple word-level accuracy
        denoised_words = denoised.lower().split()
        gold_words = gold_standard.lower().split()
        
        if not gold_words:
            return 1.0 if not denoised_words else 0.0
            
        # Calculate word-level precision and recall
        denoised_set = set(denoised_words)
        gold_set = set(gold_words)
        
        if not denoised_set and not gold_set:
            return 1.0
        if not denoised_set:
            return 0.0
        if not gold_set:
            return 0.0
            
        precision = len(denoised_set & gold_set) / len(denoised_set)
        recall = len(denoised_set & gold_set) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
            
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def calculate_kappa(self, annotator1_edits: List, annotator2_edits: List) -> float:
        """Calculate Cohen's kappa for inter-annotator agreement"""
        # Simplified kappa calculation for edit agreement
        if not annotator1_edits and not annotator2_edits:
            return 1.0
            
        # Convert edits to comparable format
        def edit_to_key(edit):
            if isinstance(edit, dict):
                return (edit.get('start_char', 0), edit.get('end_char', 0), edit.get('op', ''))
            return (getattr(edit, 'start_char', 0), getattr(edit, 'end_char', 0), getattr(edit, 'op', ''))
        
        edits1_keys = set(edit_to_key(e) for e in annotator1_edits)
        edits2_keys = set(edit_to_key(e) for e in annotator2_edits)
        
        # Calculate agreement
        agreement = len(edits1_keys & edits2_keys)
        total = len(edits1_keys | edits2_keys)
        
        if total == 0:
            return 1.0
            
        observed_agreement = agreement / total
        
        # Expected agreement (simplified)
        expected_agreement = 0.5  # Simplified assumption
        
        if expected_agreement >= 1.0:
            return 1.0
            
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        return max(0.0, min(1.0, kappa))
    
    def evaluate_all(self, original: str, denoised: str, gold_standard: str = None, 
                    medical_terms: set = None, annotator_edits: List = None) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {
            "consistency": self.calculate_consistency(original, denoised),
        }
        
        # 使用医学词典管理器或传入的术语集合
        if self.medical_terms_manager and self.medical_terms_manager._loaded:
            metrics["medical_accuracy"] = self.calculate_medical_accuracy(denoised)
        elif medical_terms:
            metrics["medical_accuracy"] = self.calculate_medical_accuracy(denoised, medical_terms)
        
        if gold_standard:
            metrics["correctness"] = self.calculate_correctness(denoised, gold_standard)
        
        if annotator_edits:
            # Simplified: assume annotator_edits is a list of [annotator1_edits, annotator2_edits]
            if len(annotator_edits) >= 2:
                metrics["kappa"] = self.calculate_kappa(annotator_edits[0], annotator_edits[1])
        
        return metrics

# -------------------------- Main DEA Pipeline --------------------------
class DetectorEditorArbiter:
    """Main DEA (Detector-Editor-Arbiter) Agent for Medical Dialogue Denoising"""
    
    def __init__(self, medical_dictionary_path: str = None, api_key: str = None):
        """
        初始化医疗对话去噪Agent
        
        Args:
            medical_dictionary_path: 医学词典文件路径
            api_key: OpenAI API密钥（可选）
        """
        # 初始化医学词典管理器
        self.medical_terms_manager = MedicalTermsManager(medical_dictionary_path)
        if medical_dictionary_path:
            self.medical_terms_manager.load_medical_dictionary()
        
        # Detector modules
        self.gec = GECTagger()
        self.spell = SpellChecker(medical_terms_manager=self.medical_terms_manager)
        self.repetition = RepetitionDetector()
        self.combined_medical = CombinedMedicalDetector(api_key) if api_key else None
        
        # Editor pipeline
        self.editor = EditorPipeline(api_key=api_key)
        
        # Arbiter pipeline
        self.arbiter = ArbiterPipeline(api_key)
        
        # Evaluation
        self.evaluator = EvaluationMetrics(medical_terms_manager=self.medical_terms_manager)
        
        # 去噪质量评分器（GEval）
        self.quality_evaluator = DenoisingQualityGEval(api_key=api_key) if api_key else None
        
        # 评分阈值
        self.quality_thresholds = {
            "accuracy": 4.2,
            "integrity": 4.5,
            "smoothness": 3.9
        }
    
    def detect_errors(self, text: str) -> List[SpanEdit]:
        """检测阶段：运行所有检测器模块（按固定顺序：spell, repetition, grammar, ambiguity, nonmedical）"""
        all_edits = []
        
        # Run all detectors in fixed order: spell, repetition, grammar, ambiguity, nonmedical
        detectors = [
            ("Spell", self.spell),           # 1. 拼写检查
            ("Repetition", self.repetition), # 2. 重复检测
            ("GEC", self.gec),               # 3. 语法纠错
        ]
        
        if self.combined_medical:
            detectors.append(("CombinedMedical", self.combined_medical))  # 4. 歧义检测(AMB) + 5. 非医患对话(NOS)
        
        for name, detector in detectors:
            try:
                edits = detector.detect(text)
                for edit in edits:
                    edit.detector_name = name
                all_edits.extend(edits)
                
                # 分别显示AMB和NOS编辑
                if name == "CombinedMedical":
                    amb_edits = [e for e in edits if e.tag == "AMB"]
                    nos_edits = [e for e in edits if e.tag == "NOS"]
                    if amb_edits:
                        print(f"[AMB] Found {len(amb_edits)} edits")
                    if nos_edits:
                        print(f"[NOS] Found {len(nos_edits)} edits")
                else:
                    print(f"[{name}] Found {len(edits)} edits")
            except Exception as e:
                print(f"[{name}] Error: {e}")
        
        return all_edits
    
    def edit_candidates(self, edits: List[SpanEdit], text: str) -> Dict:
        """编辑阶段：处理候选编辑"""
        return self.editor.run(edits, text)
    
    def arbitrate_decisions(self, edits: List[SpanEdit], original_text: str, editor_processed_text: str) -> Dict:
        """仲裁阶段：最终决策和冲突解决"""
        return self.arbiter.run(edits, original_text, editor_processed_text)
    
    def reprocess_with_llm(self, original_text: str, previous_result: str, previous_scores: Dict[str, float], 
                          arbiter_input_text: str) -> str:
        """
        使用LLM重新处理文本
        
        Args:
            original_text: 最初输入整个系统的句子（没有经过detector和editor处理的句子）
            previous_result: 上一轮的输出结果
            previous_scores: 上一轮的评分
            arbiter_input_text: 输入arbiter的句子（经过detector和editor处理后的文本）
            
        Returns:
            str: 重新处理后的文本
        """
        if not self.arbiter.api_key:
            return previous_result
        
        from openai import OpenAI
        client = OpenAI(
            api_key=self.arbiter.api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        system_prompt = """You are a medical text denoising expert. Your task is to improve the denoised medical dialogue text based on quality scores and comparison with the original text.

Your goal is to:
1. Maintain medical accuracy while improving the text quality
2. Preserve all important medical information from the original text
3. Make the text more fluent and natural
4. Address the issues identified in the quality scores"""
        
        user_prompt = f"""原始文本（未处理的输入）: {original_text}

上一轮去噪结果: {previous_result}

上一轮评分:
- 准确性 (Accuracy): {previous_scores.get('accuracy', 0):.2f}
- 完整性 (Integrity): {previous_scores.get('integrity', 0):.2f}
- 流畅性 (Smoothness): {previous_scores.get('smoothness', 0):.2f}

输入Arbiter的文本（经过Detector和Editor处理）: {arbiter_input_text}

请基于以上信息，重新处理并生成改进的去噪文本。要求：
1. 保持医学信息的准确性
2. 确保完整性，不丢失重要信息
3. 提高流畅性，使文本更自然
4. 直接输出改进后的文本，不要添加任何解释或说明"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # 清理结果，移除可能的解释性文本
            if "改进后的文本:" in result:
                result = result.split("改进后的文本:")[-1].strip()
            if "去噪后的文本:" in result:
                result = result.split("去噪后的文本:")[-1].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            return result
        except Exception as e:
            print(f"[Reprocess ERROR] {e}")
            return previous_result
    
    def denoise(self, text: str, gold_standard: str = None, verbose: bool = True) -> Dict:
        """
        医疗对话去噪主方法（包含去噪质量评分和重新处理逻辑）
        
        Args:
            text: 需要去噪的医疗对话文本（最初输入整个系统的句子）
            gold_standard: 金标准文本（可选，用于评估）
            verbose: 是否显示详细处理过程
            
        Returns:
            Dict: 包含去噪结果和评估指标的字典
        """
        # 保存原始输入文本（未经过detector和editor处理的）
        original_input_text = text
        
        if verbose:
            print("=" * 50)
            print("Medical Dialogue Denoising Agent")
            print("=" * 50)
        
        # 1. Detector stage
        if verbose:
            print("\n1. DETECTOR STAGE")
            print("-" * 30)
        detector_edits = self.detect_errors(text)
        if verbose:
            print(f"Total edits detected: {len(detector_edits)}")
        
        # 2. Editor stage  
        if verbose:
            print("\n2. EDITOR STAGE")
            print("-" * 30)
        editor_result = self.edit_candidates(detector_edits, text)
        if verbose:
            print(f"Deterministic edits: {len(editor_result['deterministic_edits'])}")
            print(f"Candidate edits: {len(editor_result['candidate_edits'])}")
        
        # 保存输入Arbiter的文本（经过Detector和Editor处理后的文本）
        # editor_result中的edited_text是应用了deterministic edits后的文本
        arbiter_input_text = editor_result.get('edited_text', text)
        if not arbiter_input_text:
            arbiter_input_text = text
        
        # 3. Arbiter stage
        if verbose:
            print("\n3. ARBITER STAGE") 
            print("-" * 30)
        all_processed_edits = editor_result['processed_edits']
        arbiter_result = self.arbitrate_decisions(all_processed_edits, original_input_text, arbiter_input_text)
        if verbose:
            print(f"Conflicts detected: {len(arbiter_result['conflicts'])}")
            print(f"Final edits applied: {len(arbiter_result['resolved_edits'])}")
        
        # 4. 去噪质量评分（在Arbiter输出完整句子后，输入LLM产生回复前）
        if verbose:
            print("\n4. DENOISING QUALITY EVALUATION STAGE")
            print("-" * 30)
        
        final_text = arbiter_result['final_text']
        all_results = []  # 存储所有轮次的结果和评分
        
        # 第一轮：对Arbiter输出进行评分
        if self.quality_evaluator:
            quality_scores = self.quality_evaluator.evaluate(original_input_text, final_text)
        else:
            quality_scores = {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
        
        all_results.append({
            "result": final_text,
            "scores": quality_scores.copy(),
            "round": 1
        })
        
        if verbose:
            print(f"Quality Scores (Round 1):")
            print(f"  Accuracy: {quality_scores.get('accuracy', 0):.2f} (threshold: {self.quality_thresholds['accuracy']})")
            print(f"  Integrity: {quality_scores.get('integrity', 0):.2f} (threshold: {self.quality_thresholds['integrity']})")
            print(f"  Smoothness: {quality_scores.get('smoothness', 0):.2f} (threshold: {self.quality_thresholds['smoothness']})")
        
        # 检查是否满足阈值要求
        meets_threshold = (
            quality_scores.get('accuracy', 0) >= self.quality_thresholds['accuracy'] and
            quality_scores.get('integrity', 0) >= self.quality_thresholds['integrity'] and
            quality_scores.get('smoothness', 0) >= self.quality_thresholds['smoothness']
        )
        
        # 如果不满足阈值，进行重新处理（最多3轮）
        max_rounds = 3
        current_round = 1
        
        while not meets_threshold and current_round < max_rounds:
            current_round += 1
            if verbose:
                print(f"\n重新处理 (Round {current_round})...")
            
            # 使用LLM重新处理
            reprocessed_text = self.reprocess_with_llm(
                original_text=original_input_text,
                previous_result=final_text,
                previous_scores=quality_scores,
                arbiter_input_text=arbiter_input_text
            )
            
            # 对重新处理的结果进行评分
            if self.quality_evaluator:
                quality_scores = self.quality_evaluator.evaluate(original_input_text, reprocessed_text)
            else:
                quality_scores = {"accuracy": 4.0, "integrity": 4.0, "smoothness": 4.0}
            
            all_results.append({
                "result": reprocessed_text,
                "scores": quality_scores.copy(),
                "round": current_round
            })
            
            if verbose:
                print(f"Quality Scores (Round {current_round}):")
                print(f"  Accuracy: {quality_scores.get('accuracy', 0):.2f} (threshold: {self.quality_thresholds['accuracy']})")
                print(f"  Integrity: {quality_scores.get('integrity', 0):.2f} (threshold: {self.quality_thresholds['integrity']})")
                print(f"  Smoothness: {quality_scores.get('smoothness', 0):.2f} (threshold: {self.quality_thresholds['smoothness']})")
            
            # 更新final_text
            final_text = reprocessed_text
            
            # 再次检查是否满足阈值
            meets_threshold = (
                quality_scores.get('accuracy', 0) >= self.quality_thresholds['accuracy'] and
                quality_scores.get('integrity', 0) >= self.quality_thresholds['integrity'] and
                quality_scores.get('smoothness', 0) >= self.quality_thresholds['smoothness']
            )
        
        # 如果三轮后仍未达到阈值，选择最佳结果
        if not meets_threshold and len(all_results) >= 3:
            if verbose:
                print(f"\n三轮处理后仍未达到阈值，选择最佳结果...")
            
            # 选择策略：accuracy最高的，其次integrity最高，integrity相同才选择smoothness最高
            best_result = all_results[0]
            for result in all_results[1:]:
                current_scores = result["scores"]
                best_scores = best_result["scores"]
                
                # 首先比较accuracy
                if current_scores.get('accuracy', 0) > best_scores.get('accuracy', 0):
                    best_result = result
                elif current_scores.get('accuracy', 0) == best_scores.get('accuracy', 0):
                    # accuracy相同时，比较integrity
                    if current_scores.get('integrity', 0) > best_scores.get('integrity', 0):
                        best_result = result
                    elif current_scores.get('integrity', 0) == best_scores.get('integrity', 0):
                        # integrity也相同时，比较smoothness
                        if current_scores.get('smoothness', 0) > best_scores.get('smoothness', 0):
                            best_result = result
            
            final_text = best_result["result"]
            quality_scores = best_result["scores"]
            
            if verbose:
                print(f"选择 Round {best_result['round']} 的结果（最佳评分）")
        
        # 5. 计算医学术语保留率（当三个指标满足阈值后）
        medical_term_retention_rate = None
        if meets_threshold or len(all_results) >= 3:
            # 三个指标满足阈值后，计算医学术语保留率
            medical_term_retention_rate = self.evaluator.calculate_medical_term_retention_rate(
                original_input_text, final_text
            )
            if verbose:
                print(f"\n医学术语保留率: {medical_term_retention_rate:.3f}")
                if self.evaluator.medical_terms_manager and self.evaluator.medical_terms_manager._loaded:
                    medical_terms = self.evaluator.medical_terms_manager.get_medical_terms()
                    original_count = self.evaluator._count_preserved_medical_terms(original_input_text, medical_terms)
                    denoised_count = self.evaluator._count_preserved_medical_terms(final_text, medical_terms)
                    print(f"  原始文本医学术语数量: {original_count}")
                    print(f"  去噪后文本医学术语数量: {denoised_count}")
                else:
                    print("  医学术语词典未加载")
        
        # 6. 传统评估（保留原有评估逻辑）
        if verbose:
            print("\n6. TRADITIONAL EVALUATION STAGE")
            print("-" * 30)
        evaluation = self.evaluator.evaluate_all(
            original=text,
            denoised=final_text,
            gold_standard=gold_standard
        )
        
        if verbose:
            print("Evaluation Metrics:")
            for metric, value in evaluation.items():
                print(f"  {metric}: {value:.3f}")

        return {
            "original_text": text,
            "detector_edits": detector_edits,
            "editor_result": editor_result,
            "arbiter_result": arbiter_result,
            "final_text": final_text,
            "evaluation": evaluation,
            "quality_scores": quality_scores,  # 去噪质量评分
            "quality_evaluation_rounds": len(all_results),  # 评分轮次
            "all_quality_results": all_results,  # 所有轮次的结果
            "medical_term_retention_rate": medical_term_retention_rate  # 医学术语保留率
        }
    
    def quick_denoise(self, text: str) -> str:
        """
        快速去噪方法，只返回去噪后的文本
        
        Args:
            text: 需要去噪的医疗对话文本
            
        Returns:
            str: 去噪后的文本
        """
        result = self.denoise(text, verbose=False)
        return result['final_text']
    
    def batch_denoise(self, texts: List[str], verbose: bool = True) -> List[Dict]:
        """
        批量去噪方法
        
        Args:
            texts: 需要去噪的文本列表
            verbose: 是否显示处理过程
            
        Returns:
            List[Dict]: 每个文本的去噪结果列表
        """
        results = []
        for i, text in enumerate(texts):
            if verbose:
                print(f"\n处理第 {i+1}/{len(texts)} 个文本...")
            result = self.denoise(text, verbose=False)
            results.append(result)
        return results

# 保持向后兼容的别名
DEAPipeline = DetectorEditorArbiter

# -------------------------- 配置说明 --------------------------
"""
使用前需要配置的参数：

1. 模型路径配置：
   - GECTagger: 语法纠错模型路径
   - AmbiguityDetector: GlossBERT模型路径  
   - FragmentDetector: SimCSE模型路径

2. API密钥配置：
   - OpenAI API密钥用于LLM功能
   - 替换代码中的硬编码密钥

3. 医学词典配置：
   - 医学术语白名单
   - 歧义词义词典
   - 医学词汇表

4. 可选参数：
   - medical_terms: 医学术语集合
   - 各种检测器的阈值参数
   - 评分权重参数
"""
    

# 保持向后兼容的别名
DEAPipeline = DetectorEditorArbiter

# 导出主要类，方便外部调用
__all__ = [
    'DetectorEditorArbiter',
    'DEAPipeline', 
    'SpanEdit',
    'BaseExtractorModule',
    'GECTagger',
    'SpellChecker', 
    'RepetitionDetector',
    'CombinedMedicalDetector',
    'EditorPipeline',
    'ArbiterPipeline',
    'EvaluationMetrics',
    'MedicalTermsManager'
]
