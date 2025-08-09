# -*- coding: utf-8 -*-
"""
generate_lyrics (improved)
- MeCab (fugashi + unidic-lite) でモーラ数カウント
- 違反行のみ一括再生成（API回数削減）
- 厳格JSON出力を強制するプロンプト
- レスポンスの自動JSON抽出/検証
- デバッグログ一式の保存

準備:
  sudo apt install mecab libmecab-dev
  pip install fugashi[unidic-lite] google-generativeai

Gemini:
  無料枠は gemini-1.5-flash-latest を推奨。
  APIキーは環境変数 GEMINI_API_KEY に置くか、下の API_KEY を直接設定。
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
    from fugashi import GenericTagger
except Exception as e:
    raise RuntimeError("fugashi (with unidic-lite) が必要です。pip install fugashi[unidic-lite]") from e


# ============ 設定 ============

# ここに直接書くか、環境変数 GEMINI_API_KEY を使ってください。
API_KEY = os.getenv("GEMINI_API_KEY", "あなたのGEMINI_API_KEY").strip()
MODEL_NAME = "gemini-1.5-flash-latest"  # 無料枠向け。必要に応じて変更可。
MAX_ITERS = 10                           # 修正ループの最大試行回数（無料枠向けに控えめ）
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


# ============ モーラカウント（MeCab + 独自規則） ============

SMALL_KATAKANA = set("ァィゥェォャュョヮ")
MORAIC_CHARS = set("ーッ")

@dataclass
class TokenInfo:
    surface: str
    reading: str
    mora: int

import unidic_lite
import os
from fugashi import Tagger

class MoraCounter:
    def __init__(self):
        dicdir = unidic_lite.DICDIR
        rcfile = os.path.join(dicdir, 'mecabrc')
        try:
            self.tagger = Tagger(f'-r {rcfile} -d {dicdir}')
        except Exception as e:
            raise RuntimeError(f"Tagger の初期化に失敗しました: {e}")

    def _to_katakana(self, s: str) -> str:
        out = []
        for ch in s:
            code = ord(ch)
            if 0x3041 <= code <= 0x3096:  # ひらがな→カタカナ
                out.append(chr(code + 0x60))
            else:
                out.append(ch)
        return "".join(out)

    def _katakana_mora_count(self, s: str) -> int:
        cnt = 0
        for ch in s:
            if ch in SMALL_KATAKANA:
                continue  # 小書きは直前に合流
            if ch in MORAIC_CHARS:
                cnt += 1
            elif "ァ" <= ch <= "ン":
                cnt += 1
            else:
                # 記号・英数などはモーラに含めない
                continue
        return cnt

    def analyze(self, text: str) -> Tuple[int, List[TokenInfo]]:
        total = 0
        tokens: List[TokenInfo] = []
        for word in self.tagger(text):
            reading = getattr(word.feature, "Reading", None) or word.surface
            kat = self._to_katakana(reading)
            mora = self._katakana_mora_count(kat)
            tokens.append(TokenInfo(surface=word.surface, reading=kat, mora=mora))
            total += mora
        return total, tokens


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
    return f"""あなたは厳密な出力制約に従う日本語作詞アシスタントです。以下を必ず守ってください。

- 出力は JSON のみ（説明文・前置き・コードブロックは禁止）。
- JSONスキーマ: {{"lines": [string, ...]}} のみを返す。
- 行数は正確に {n_lines} 行。
- 各行のモーラ数は上から順に [{pat}] を繰り返し適用し、厳密一致させる。
- 英数字や記号で水増ししない。意味の通る自然な日本語にする。
- 出力前に自己検証を行い、各行が指定モーラに一致していることを確認する。

テーマ: 「{theme}」
雰囲気: 「{mood}」

出力例（例の内容は無関係。形式のみ参照）:
{json.dumps(example, ensure_ascii=False)}

今すぐ JSON のみを返してください。"""

def build_regen_prompt(theme: str, mood: str, fixed: Dict[int, str], to_fix: Dict[int, int]) -> str:
    # fixed は変更禁止プレビュー、to_fix は 0-based index: 要求モーラ
    fixed_preview = "\n".join(f"{i+1}: {txt}" for i, txt in sorted(fixed.items()))
    target_list = "\n".join(f"{i+1}: 要求モーラ={m}" for i, m in sorted(to_fix.items()))
    example = {"replacements": {"3": "（3行目の新しい歌詞）", "7": "（7行目の新しい歌詞）"}}
    return f"""あなたは部分修正に厳密に従う日本語作詞アシスタントです。以下を必ず守ってください。

- 出力は JSON のみ（説明・前置き・コードブロックは禁止）。
- JSONスキーマ: {{"replacements": {{"LINE_NUMBER(1-based)": "string"}}}}
- 返すキーは「修正対象行のみ」。指定されていない行は一切変更禁止。
- 各修正行は指定モーラ数に厳密一致させる。
- 出力前に自己検証を行う。

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
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
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
        # 最後の手段として ' を " に置換（強引）
        blk2 = re.sub(r"'", '"', blk)
        return json.loads(blk2)


# ============ Gemini クライアント ============

class GeminiClient:
    def __init__(self, model_name: str, logger: logging.Logger, run_dir: Path):
        self.model = genai.GenerativeModel(model_name)
        self.logger = logger
        self.run_dir = run_dir

    def send(self, prompt: str, tag: str) -> str:
        # プロンプト/応答をログに保存
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


# ============ メイン処理（公開API互換） ============

def configure_gemini() -> bool:
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        print("--- エラー ---")
        print(f"APIキーの設定に失敗しました: {e}")
        print("環境変数 GEMINI_API_KEY またはスクリプト上部の API_KEY を確認してください。")
        return False

def generate_lyrics_with_mora_check(theme: str, mood: str, mora_structure: List[int], verses: int) -> Optional[str]:
    """
    指定テーマとモーラ構成で歌詞を生成。違反行のみを一括再生成しながらモーラ一致まで修正。
    返り値: 最終歌詞（番区切りの空行入り）/ 失敗時 None
    """
    if not configure_gemini() or API_KEY in ("YOUR_API_KEY", "あなたのAPIキー"):
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

        # ログ保存（違反詳細）
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
            print("\n--- 全ての行のモーラ数が一致しました！ ---")
            break

        # 修正対象を集約（0-based index）
        to_fix: Dict[int, int] = {v.index: v.want_mora for v in violations}
        fixed: Dict[int, str] = {i: txt for i, txt in enumerate(lines) if i not in to_fix}

        print(f"\n【試行回数: {it}/{MAX_ITERS}】違反 {len(to_fix)} 行 → 一括修正リクエスト")
        logger.info("iteration %d: fix %d lines", it, len(to_fix))

        # 一括再生成
        try:
            replacements = client.regenerate_lines(theme, mood, fixed, to_fix, it)
        except Exception as e:
            logger.error("部分修正失敗: %s", e)
            print("--- エラー ---")
            print(f"Geminiの部分修正でエラー: {e}")
            # 無料枠配慮で少し待機して続行（スキップせず粘る）
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

        # 無料枠配慮の軽いレート制御
        time.sleep(0.8)

        if it == MAX_ITERS:
            print("\n--- 最大試行回数に達しました ---")
            print("指定モーラ数に完全一致しない行が残っています。最終結果と違反一覧を logs に保存しました。")

    # 最終整形と出力
    final_violations = check_mora_constraints(lines, mora_structure, counter)
    save_json({
        "final_lines": lines,
        "final_violations": [
            {"index": v.index, "want": v.want_mora, "got": v.got_mora, "text": v.text} for v in final_violations
        ]
    }, run_dir / "final.json")

    # 1番ごとに空行で区切る
    out = []
    for i, line in enumerate(lines, 1):
        out.append(line)
        if i % len(mora_structure) == 0:
            out.append("")  # 空行
    final_lyrics = "\n".join(out).rstrip() + "\n"

    print(f"\nログ保存先: {run_dir}")
    if final_violations:
        print(f"未解決の違反行: {len(final_violations)}（logsで確認可）")
    return final_lyrics


# ============ スクリプトとして実行 ============

if __name__ == "__main__":
    # 任意でここを書き換えて実行してください
    THEME = "恋愛"
    MOOD = "少し悲しい感じ"
    MORA_STRUCTURE = [7, 7, 8, 8]
    VERSES = 4

    result = generate_lyrics_with_mora_check(THEME, MOOD, MORA_STRUCTURE, VERSES)
    if result:
        print("\n完成した歌詞:")
        print("=" * 30)
        print(result)
        print("=" * 30)