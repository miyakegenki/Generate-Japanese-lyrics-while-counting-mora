# -*- coding: utf-8 -*-
"""
generate_lyrics_ver10 (fixed)
- MeCab (fugashi + unidic-lite) でモーラ数カウント
- 拗音・促音・長音の正確な処理
- 漢字→カナ変換の強化（cutletライブラリ使用）
- 違反行のみ一括再生成（API回数削減）
- 厳格JSON出力を強制するプロンプト
- レスポンスの自動JSON抽出/検証
- デバッグログ一式の保存

準備:
  pip install fugashi[unidic-lite] google-generativeai cutlet

Gemini:
  無料枠は gemini-1.5-flash-latest を推奨。
  APIキーは環境変数 GEMINI_API_KEY に置く。
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai

try:
    from fugashi import Tagger
    import unidic_lite
except Exception as e:
    raise RuntimeError("fugashi (with unidic-lite) が必要です。pip install fugashi[unidic-lite]") from e

try:
    import cutlet
except Exception as e:
    raise RuntimeError("cutlet が必要です。pip install cutlet") from e


# ============ 設定 ============

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest"
MAX_ITERS = 10
LOG_BASE_DIR = "logs"


# ============ ログ/保存ユーティリティ ============

def setup_run_dir(base: str = LOG_BASE_DIR) -> Path:
    base_p = Path(base)
    base_p.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_p / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("lyrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger

def save_json(obj: Any, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ============ モーラカウント（改善版） ============

@dataclass
class TokenInfo:
    surface: str
    reading: str
    mora: int

class MoraCounter:
    """
    日本語テキストのモーラ数を正確にカウントするクラス
    - cutletで漢字→カナ変換を確実に実行
    - 拗音・促音・長音を正確に処理
    """
    def __init__(self):
        # cutletの初期化（ローマ字化せず、カタカナ読みを取得）
        self.katsu = cutlet.Cutlet()
        self.katsu.use_foreign_spelling = False
        
    def _get_katakana_reading(self, text: str) -> str:
        """
        テキストをカタカナ読みに変換
        cutletは内部でMeCabを使い、確実に読み仮名を取得できる
        """
        try:
            # cutletのromaji()を使ってローマ字化してから、
            # カタカナに戻すのではなく、直接カタカナを取得
            # 注: cutletは本来ローマ字化用だが、内部の読み情報を活用
            
            # 代替: ひらがな→カタカナ変換を自前で実施
            reading = self.katsu.romaji(text, capitalize=False)
            # ローマ字からカタカナへ（簡易的な逆変換）
            # より確実な方法: MeCabの読み仮名を直接取得
            return self._text_to_katakana_direct(text)
        except:
            return self._text_to_katakana_direct(text)
    
    def _text_to_katakana_direct(self, text: str) -> str:
        """
        MeCabの読み仮名を使ってカタカナに変換
        cutletの内部実装を参考に、確実に読みを取得
        """
        result = []
        for word in self.katsu.tagger(text):
            # MeCabのfeatureから読み仮名を取得
            features = word.feature
            reading = features.kana if hasattr(features, 'kana') else None
            
            if reading and reading != '*':
                # 読み仮名が取得できた場合
                result.append(reading)
            else:
                # 読み仮名がない場合は表層形をカタカナ化
                result.append(self._hiragana_to_katakana(word.surface))
        
        return ''.join(result)
    
    def _hiragana_to_katakana(self, text: str) -> str:
        """ひらがな→カタカナ変換"""
        result = []
        for ch in text:
            code = ord(ch)
            if 0x3041 <= code <= 0x3096:  # ひらがな範囲
                result.append(chr(code + 0x60))  # カタカナに変換
            else:
                result.append(ch)
        return ''.join(result)
    
    def _count_mora_from_katakana(self, katakana: str) -> int:
        """
        カタカナ文字列からモーラ数をカウント
        
        ルール:
        - 通常のカタカナ: 1モーラ
        - 小書き文字（ャュョァィゥェォヮ）: 直前の文字と結合（0モーラ扱い）
        - 促音「ッ」: 1モーラ
        - 長音「ー」: 1モーラ
        - 撥音「ン」: 1モーラ
        """
        small_kana = set('ァィゥェォヮヵヶャュョ')  # 小書き文字
        
        mora_count = 0
        for ch in katakana:
            if ch in small_kana:
                # 小書き文字は拗音として直前に合流（カウントしない）
                continue
            elif ch in 'ーッ':
                # 長音・促音は1モーラ
                mora_count += 1
            elif 'ァ' <= ch <= 'ヶ':
                # 通常のカタカナ
                mora_count += 1
            # それ以外（記号・空白など）はカウントしない
        
        return mora_count
    
    def analyze(self, text: str) -> Tuple[int, List[TokenInfo]]:
        """
        テキストを形態素解析し、各トークンとモーラ数を返す
        """
        tokens: List[TokenInfo] = []
        total_mora = 0
        
        for word in self.katsu.tagger(text):
            surface = word.surface
            
            # 読み仮名の取得（確実性を高める）
            features = word.feature
            if hasattr(features, 'kana') and features.kana and features.kana != '*':
                reading = features.kana
            elif hasattr(features, 'pron') and features.pron and features.pron != '*':
                reading = features.pron
            else:
                # fallback: 表層形をカタカナ化
                reading = self._hiragana_to_katakana(surface)
            
            # モーラ数をカウント
            mora = self._count_mora_from_katakana(reading)
            
            tokens.append(TokenInfo(
                surface=surface,
                reading=reading,
                mora=mora
            ))
            total_mora += mora
        
        return total_mora, tokens


# ============ 制約チェック ============

@dataclass
class LineViolation:
    index: int
    want_mora: int
    got_mora: int
    text: str
    details: List[TokenInfo]

def check_mora_constraints(lines: List[str], pattern: List[int], counter: MoraCounter) -> List[LineViolation]:
    v: List[LineViolation] = []
    for i, line in enumerate(lines):
        want = pattern[i % len(pattern)]
        got, tokens = counter.analyze(line)
        if got != want:
            v.append(LineViolation(index=i, want_mora=want, got_mora=got, text=line, details=tokens))
    return v


# ============ プロンプト（初回/部分修正） ============

def build_initial_prompt(theme: str, mood: str, n_lines: int, pattern: List[int]) -> str:
    pat = ",".join(str(x) for x in pattern)
    example = {"lines": ["（ここに1行目）", "（ここに2行目）"]}
    
    # モーラ数の例を具体的に示す
    mora_examples = []
    for m in pattern[:2]:  # 最初の2つのパターンを例示
        if m == 7:
            mora_examples.append(f"7モーラの例: 「さくらのはなが」(サクラノハナガ = サ・ク・ラ・ノ・ハ・ナ・ガ)")
        elif m == 5:
            mora_examples.append(f"5モーラの例: 「はるのかぜ」(ハルノカゼ = ハ・ル・ノ・カ・ゼ)")
        elif m == 8:
            mora_examples.append(f"8モーラの例: 「あおいそらをみる」(アオイソラヲミル = ア・オ・イ・ソ・ラ・ヲ・ミ・ル)")
    
    mora_guide = "\n".join(mora_examples) if mora_examples else ""
    
    return f"""あなたは厳密な出力制約に従う日本語作詞アシスタントです。以下を必ず守ってください。

- 出力は JSON のみ（説明文・前置き・コードブロックは禁止）。
- JSONスキーマ: {{"lines": [string, ...]}} のみを返す。
- 行数は正確に {n_lines} 行。
- 各行のモーラ数は上から順に [{pat}] を繰り返し適用し、厳密一致させる。

【モーラ数とは】
日本語の音の単位です。「きょう」は2文字ですが2モーラ(キョ・ウ)、「がっこう」は4文字ですが4モーラ(ガ・ッ・コ・ウ)です。
{mora_guide}

【重要な注意】
- 小書き文字（ゃゅょ）は直前の文字と合わせて1モーラ
- 促音「っ」は1モーラ
- 長音「ー」は1モーラ
- 撥音「ん」は1モーラ
- 出力前に必ず各行を音読して、指定モーラ数と一致することを確認する

テーマ: 「{theme}」
雰囲気: 「{mood}」

出力例（例の内容は無関係。形式のみ参照）:
{json.dumps(example, ensure_ascii=False)}

今すぐ JSON のみを返してください。"""

def build_regen_prompt(theme: str, mood: str, fixed: Dict[int, str], to_fix: Dict[int, int]) -> str:
    fixed_preview = "\n".join(f"{i+1}: {txt}" for i, txt in sorted(fixed.items()))
    target_list = "\n".join(f"{i+1}: 要求モーラ={m}" for i, m in sorted(to_fix.items()))
    example = {"replacements": {"3": "（3行目の新しい歌詞）", "7": "（7行目の新しい歌詞）"}}
    
    return f"""あなたは部分修正に厳密に従う日本語作詞アシスタントです。以下を必ず守ってください。

- 出力は JSON のみ（説明・前置き・コードブロックは禁止）。
- JSONスキーマ: {{"replacements": {{"LINE_NUMBER(1-based)": "string"}}}}
- 返すキーは「修正対象行のみ」。指定されていない行は一切変更禁止。
- 各修正行は指定モーラ数に厳密一致させる。

【モーラ数の数え方】
音読して数えます。「きょう」=2モーラ、「がっこう」=4モーラ、「さくら」=3モーラ

テーマ: 「{theme}」
雰囲気: 「{mood}」

変更禁止（プレビュー）:
{fixed_preview}

修正対象（1-based行番号と要求モーラ）:
{target_list}

出力例（形式のみ参照）:
{json.dumps(example, ensure_ascii=False)}

今すぐ JSON のみを返してください。"""


# ============ レスポンスのJSON抽出/検証 ============

def extract_json_block(text: str) -> Optional[str]:
    # ```json ... ``` 優先
    m = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for blk in m:
        blk = blk.strip()
        if blk.startswith("{") or blk.startswith("["):
            return blk
    # 全体がJSON
    t = text.strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t
    # 最初の { から対応 } まで
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None

def sanitize_json(s: str) -> str:
    s = s.replace(""", '"').replace(""", '"').replace("'", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def parse_model_json(text: str) -> Any:
    blk = extract_json_block(text)
    if blk is None:
        raise ValueError("モデル応答からJSONを抽出できませんでした。")
    blk = sanitize_json(blk)
    try:
        return json.loads(blk)
    except json.JSONDecodeError:
        blk2 = re.sub(r"'", '"', blk)
        return json.loads(blk2)


# ============ Gemini クライアント ============

class GeminiClient:
    def __init__(self, model_name: str, logger: logging.Logger, run_dir: Path):
        self.model = genai.GenerativeModel(model_name)
        self.logger = logger
        self.run_dir = run_dir

    def send(self, prompt: str, tag: str) -> str:
        (self.run_dir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
        resp = self.model.generate_content(prompt)
        text = resp.text or ""
        (self.run_dir / f"{tag}_response.txt").write_text(text, encoding="utf-8")
        return text

    def generate_initial_lines(self, theme: str, mood: str, n_lines: int, pattern: List[int]) -> List[str]:
        prompt = build_initial_prompt(theme, mood, n_lines, pattern)
        raw = self.send(prompt, "step_00_initial")
        data = parse_model_json(raw)
        if not isinstance(data, dict) or "lines" not in data or not isinstance(data["lines"], list):
            raise ValueError("初回応答が 'lines' 配列を含むJSONではありません。")
        lines = [str(x).strip() for x in data["lines"]]
        if len(lines) != n_lines:
            raise ValueError(f"行数が不一致: 期待 {n_lines}, 実際 {len(lines)}")
        return lines

    def regenerate_lines(self, theme: str, mood: str, fixed: Dict[int, str], to_fix: Dict[int, int], iter_idx: int) -> Dict[int, str]:
        prompt = build_regen_prompt(theme, mood, fixed, to_fix)
        raw = self.send(prompt, f"step_{iter_idx:02d}_regen")
        data = parse_model_json(raw)
        if not isinstance(data, dict) or "replacements" not in data or not isinstance(data["replacements"], dict):
            raise ValueError("修正応答が 'replacements' マップを含むJSONではありません。")
        out: Dict[int, str] = {}
        for k, v in data["replacements"].items():
            try:
                idx0 = int(k) - 1
            except Exception:
                continue
            out[idx0] = str(v).strip()
        return out


# ============ メイン処理 ============

def configure_gemini() -> bool:
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        print("--- エラー ---")
        print(f"APIキーの設定に失敗しました: {e}")
        print("環境変数 GEMINI_API_KEY を確認してください。")
        return False

def generate_lyrics_with_mora_check(theme: str, mood: str, mora_structure: List[int], verses: int) -> Optional[str]:
    if not API_KEY:
        print("有効なAPIキーを設定してください。")
        return None

    n_lines = len(mora_structure) * verses
    run_dir = setup_run_dir(LOG_BASE_DIR)
    logger = setup_logger(run_dir)
    save_json({
        "theme": theme,
        "mood": mood,
        "mora_structure": mora_structure,
        "verses": verses,
        "n_lines": n_lines,
        "model": MODEL_NAME,
        "max_iters": MAX_ITERS,
    }, run_dir / "config.json")

    counter = MoraCounter()
    client = GeminiClient(MODEL_NAME, logger, run_dir)

    print("--- 歌詞の生成を開始します ---")
    print(f"目標: {verses}番 × {mora_structure}モーラ構成")
    logger.info("初回生成を開始")
    
    try:
        lines = client.generate_initial_lines(theme, mood, n_lines, mora_structure)
    except Exception as e:
        logger.error("初回生成失敗: %s", e)
        print("--- エラー ---")
        print(f"Geminiの初回生成でエラー: {e}")
        return None

    # 修正ループ
    for it in range(1, MAX_ITERS + 1):
        violations = check_mora_constraints(lines, mora_structure, counter)

        # デバッグ出力: 各行のモーラ数を表示
        print(f"\n【試行 {it}】現在の状態:")
        for i, line in enumerate(lines):
            want = mora_structure[i % len(mora_structure)]
            got, tokens = counter.analyze(line)
            status = "✓" if got == want else "✗"
            print(f"  {status} {i+1}行目 [{got}/{want}モーラ]: {line}")
            if got != want:
                # 詳細表示
                token_detail = " + ".join([f"{t.surface}({t.mora})" for t in tokens])
                print(f"      分解: {token_detail}")

        # 違反詳細の保存
        vr = []
        for v in violations:
            vr.append({
                "index": v.index,
                "want": v.want_mora,
                "got": v.got_mora,
                "text": v.text,
                "tokens": [{"surface": t.surface, "reading": t.reading, "mora": t.mora} for t in v.details]
            })
        save_json({"iteration": it, "violations": vr}, run_dir / f"step_{it:02d}_violations.json")

        if not violations:
            print("\n✓✓✓ 全ての行のモーラ数が一致しました！ ✓✓✓")
            break

        # 修正対象を集約
        to_fix: Dict[int, int] = {v.index: v.want_mora for v in violations}
        fixed: Dict[int, str] = {i: txt for i, txt in enumerate(lines) if i not in to_fix}

        print(f"\n→ 違反 {len(to_fix)} 行を修正します...")
        logger.info("iteration %d: fix %d lines", it, len(to_fix))

        try:
            replacements = client.regenerate_lines(theme, mood, fixed, to_fix, it)
        except Exception as e:
            logger.error("部分修正失敗: %s", e)
            print(f"エラー: {e}")
            time.sleep(2)
            continue

        # 適用
        for idx, new_text in replacements.items():
            if idx in to_fix:
                lines[idx] = new_text

        save_json({
            "iteration": it,
            "replacements": {str(k+1): v for k, v in replacements.items()},
            "current_lines": lines
        }, run_dir / f"step_{it:02d}_replacements.json")

        time.sleep(0.8)

        if it == MAX_ITERS:
            print("\n--- 最大試行回数に達しました ---")
            final_violations = check_mora_constraints(lines, mora_structure, counter)
            if final_violations:
                print(f"未解決の違反: {len(final_violations)}行")

    # 最終結果
    final_violations = check_mora_constraints(lines, mora_structure, counter)
    save_json({
        "final_lines": lines,
        "final_violations": [
            {"index": v.index, "want": v.want_mora, "got": v.got_mora, "text": v.text} for v in final_violations
        ]
    }, run_dir / "final.json")

    # 番ごとに空行で区切る
    out = []
    for i, line in enumerate(lines, 1):
        out.append(line)
        if i % len(mora_structure) == 0:
            out.append("")
    final_lyrics = "\n".join(out).rstrip() + "\n"

    print(f"\nログ保存先: {run_dir}")
    return final_lyrics


# ============ スクリプトとして実行 ============

if __name__ == "__main__":
    if not configure_gemini():
        exit(1)
    
    # 設定（任意に変更可能）
    THEME = "何でもOK"
    MOOD = "何でもOK"
    MORA_STRUCTURE = [7, 7]
    VERSES = 1

    result = generate_lyrics_with_mora_check(THEME, MOOD, MORA_STRUCTURE, VERSES)
    if result:
        print("\n" + "="*50)
        print("完成した歌詞:")
        print("="*50)
        print(result)
        print("="*50)