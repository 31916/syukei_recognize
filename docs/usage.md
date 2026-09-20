# 実行と再現の手順

[研究概要に戻る](../README.md) · [実験の詳細](research.md) · [データ形式](data-format.md)

この手順は現在のコードを手元のデータで実行するためのものです。卒業研究の撮影動画、標本単位の予測、学習済みモデルは同梱していないため、論文の44.25%・75.09%を再現したとはいえません。

## セットアップ

Python **3.10〜3.12** を想定しています。依存ライブラリは [requirements.txt](../requirements.txt) に記載しています。MediaPipe は、この実装で使用する `mp.solutions.holistic` を利用できる [0.10.21](https://pypi.org/project/mediapipe/0.10.21/) に固定しています。

```sh
git clone https://github.com/31916/syukei_recognize.git
cd syukei_recognize
python -m venv .venv
```

仮想環境を有効にします。

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

```sh
python -m pip install -r requirements.txt
```

## 実行方法

### 1. 動画を配置する

`data/video/` に、`<被験者ID>_<手形番号>.mp4` という名前で動画を置きます。

```text
data/video/
├── subject01_01.mp4
├── subject01_02.mp4
├── subject02_01.mp4
└── ...
```

被験者 ID と手形番号には `_` を含めないでください。手形番号には元データと同じゼロ埋め表記を使います。左右の接尾辞 `r` / `l` は抽出時に自動で付加されます。

### 2. 特徴量を抽出する

```sh
python extract_hand_features.py
```

例えば `subject01_01.mp4` から、`data/output/hand_info/subject01_01r.json` と `subject01_01l.json` を作ります。検出された手がない場合は空配列になります。JSON の詳細は [データ形式](data-format.md) を参照してください。

### 3. 交差検証で学習する

```sh
python train_cross_validation.py
```

デフォルトでは 5 分割、最大 100 エポック、バッチサイズ 16 です。有効な標本を持つ被験者が **5人以上**必要です。同一被験者が学習側と検証側にまたがらないように分割します。

```sh
python train_cross_validation.py --folds 3 --epochs 50 --batch-size 16
```

各 JSON の左右それぞれから最大 5 フレームを抽出します。選んだ標本を `selected_data.json` に保存し、評価にも同じ標本・同じ順序を使用します。

### 4. 評価・集計する

```sh
python evaluate_models.py
python summarize_results.py
```

評価は `model/` 内に保存された標本・クラス順・検証インデックスを読み込みます。抽出元の JSON は読み直しません。

```text
data/output/
├── frame/                    # 元のフレーム画像
├── landmark/                 # 手の骨格を重ねた画像
├── pos/                      # 左右21点の画素座標
├── hand_info/                # 20角度、方向、左右情報
├── model/
│   ├── selected_data.json    # 学習・評価で共有する標本
│   ├── label_encoder.json    # モデル出力に対応するクラス順
│   ├── fold_info.json        # モデル名と検証標本の行番号
│   ├── run_config.json       # 学習条件
│   ├── best_models_info.json
│   └── best_model_fold_*.keras
└── evaluation/
    ├── fold_1/
    │   ├── top_10_predictions.json
    │   └── results.json
    ├── ...
    └── rl_result.json        # 左右をまとめた Top-1 正解率
```

### 保存場所を変更する

各スクリプトの `--help` で引数を確認できます。デフォルトの `data/` は**リポジトリの場所を基準**とし、明示した相対パスはコマンド実行場所を基準とします。

```sh
python extract_hand_features.py --video-dir ../videos --output-dir ../experiment
python train_cross_validation.py --data-dir ../experiment/hand_info --output-dir ../experiment/model
python evaluate_models.py --model-dir ../experiment/model --output-dir ../experiment/evaluation
python summarize_results.py --evaluation-dir ../experiment/evaluation
```

学習・評価には空の出力ディレクトリを指定します。再実験は別ディレクトリに保存してください。特徴量抽出は同名の JSON・画像を上書きしますが、以前に保存した余分な画像は削除しないため、条件を変えるときは新しい出力先を使ってください。`data/` やモデルファイルは Git の追跡対象から除外しています。

## 動作確認

```sh
python -m unittest discover -s tests -v
```

人工データで、標本の再利用、被験者分割、学習から評価・集計までの接続、手が写っていない動画の抽出処理を確認します。元の研究データでの精度再現は別途必要です。

## 再現実験で保存する情報

新しい実験では、利用したコミット、Python・依存ライブラリのバージョン、動画・ラベルの版、各ラベルの有効標本数を記録します。モデルの出力フォルダには、選択した標本、クラス順、fold の検証インデックス、学習設定が保存されます。モデルとこれらの JSON を同じ実験の一式として保管してください。

```sh
python --version
python -m pip freeze > environment.txt
git rev-parse HEAD
```

論文との比較前に、[資料と実装の相違](sources.md#implementation-differences)を確認してください。手の左右・向きの入力、評価値の集計方法、ブレの除外条件は同一とは限りません。現在の交差検証では、各 fold の検証データを最良モデルの選択にも使用しており、独立した最終テストではありません。

旧ファイル名からの対応は [移行メモ](migration.md) を参照してください。
