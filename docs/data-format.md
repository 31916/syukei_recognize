# データ形式

## 動画とラベルの命名

動画は `<subject>_<shape>.mp4`、抽出後の特徴量は `<subject>_<shape>r.json` / `<subject>_<shape>l.json` とします。例：`subject01_01.mp4` → `subject01_01r.json`。

- `subject`：被験者 ID。GroupKFold のグループに使います。
- `shape`：手形番号。`01` のように元データのゼロ埋めを維持します。
- `r` / `l`：MediaPipe の右手 / 左手。学習時には接尾辞を含めてクラス名にします。

被験者 ID と手形番号に `_` は使えません。クラス番号と実際の手形の対応表は含まれていません。

## hand_info/*.json

フレームごとの記録の配列です。次は形式説明用の人工値です。

```json
[
  {
    "frame": 12,
    "angles": [30, 150, 160, 170, 20, 150, 160, 170, 15, 150, 160, 170, 10, 150, 160, 170, 45, 150, 160, 170],
    "yaw": 1.2,
    "rl": 1
  }
]
```

| キー | 内容 |
| --- | --- |
| `frame` | 元動画の0始まりのフレーム番号 |
| `angles` | 20個の角度（度）。親指、人差し指、中指、薬指、小指の順に各4個 |
| `yaw` | 手首（点0）から人差し指先端（点8）への画像平面上の方向（ラジアン） |
| `rl` | 右手は `1`、左手は `0` |

角度に使う3点の組は `extract_hand_features.py` の `ANGLE_LANDMARK_TRIPLETS` に定義しています。`yaw` は3次元姿勢や手の表裏を表す値ではありません。学習に使うのは `angles` のみです。

未検出の手は記録しません。有効な記録がない手のファイルは `[]` です。学習時は20個の有限な数値を持つ標本だけを採用します。

## pos/*.json

各記録は `frame`、`right_hand_pos`、`left_hand_pos` を持ちます。各手の座標は MediaPipe の順序に沿った21個の `[x, y]`（画素）です。未検出の手は `null` です。両手未検出のフレームと、移動量が閾値を超えたフレームは記録しません。

角度データは移動量の判定前に保存するため、`pos/`・画像と `hand_info/` の記録数は一致するとは限りません。

## model/ 内のファイル

- `selected_data.json`：ファイル名をキーに、`right` / `left` の標本配列を保存します。**キーと配列の順序は変更しないでください。**
- `label_encoder.json`：モデルの出力列に対応するクラス名の配列です。
- `fold_info.json`：`[モデルファイル名, 検証標本の行番号の配列]` を fold 順に保存します。モデル名は同じ `model/` を基準とした相対名です。
- `run_config.json`：fold 数、エポック上限、バッチサイズ、seed、標本数、被験者数、クラス数です。
- `best_models_info.json`：各 fold の最小検証損失とモデルファイル名です。

これらと `.keras` ファイルは同じ実験の一式として扱ってください。評価は `selected_data.json` を読み、再抽出を行いません。

## evaluation/ 内のファイル

各 `fold_N/top_10_predictions.json` は `true_label` と `top_10` を持つ記録の配列です。`top_10` は `label` と `probability` の組を確率の降順で最大10件保持します。クラスが10未満なら全クラスです。値は Softmax 後の確率であり、logit ではありません。

`fold_N/results.json` は `mean_accuracy`、`variance_accuracy`、`accuracy_per_label` を持ちます。後者の各クラスは `correct`、`total`、`accuracy` を持ちます。標本がないクラスは平均・分散から外します。

`rl_result.json` は全 fold の予測を集め、左右を同一視した Top-1 正解率を保存します。旧形式を維持して、分散のキーは `variance` です。
