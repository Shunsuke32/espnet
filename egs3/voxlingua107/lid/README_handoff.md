# ESPnet3 VoxLingua107 LID 開発・移行引き継ぎ

更新日: 2026-09-19。通常の使い方は [readme.md](readme.md)、外部データの準備契約は [external_evaluation.md](external_evaluation.md) を参照。
この文書はLID追加全体の変更理由、検証範囲、別サーバーで再開する際の注意をまとめる。

## 1. 現在の状態

- 作業ツリー: `/home/mitsumori/espnet3-dev`、branch: `espnet3-dev`。
- 基準HEAD: `69f7d271eb64103bc2c505fd27da13e013e25540`。
- LID追加は `espnet3-dev` のコミット対象。上記基準HEADだけではLIDは入らないため、引き継ぎ先ではLID追加コミットを取得する。
- 今回の公開モデル検証では実装変更は不要だった。追加したのは検証用設定・成果物とこの引き継ぎ文書。
- 既存ASR、共通training/Lightning/collect_stats、ESPnet2には現在のtracked差分なし。
- 共通部分の既存差分は `espnet3/components/data/dataloader.py` と対応テストの2ファイル。影響範囲は後述。

## 2. 設計方針と変更箇所

「設定で指定できることは設定へ」「モデル・sampler・評価処理は既存ESPnet2/3を再利用」「LID固有処理はLID側へ」を基本にした。
ASRと同じくSystemはステージの窓口、TaskはESPnet2モデル構築との橋渡し、DatasetBuilder/Datasetはコーパス固有の準備・読込を担当する。

| 場所（リポジトリrootから） | 追加・変更内容と理由 |
| --- | --- |
| `espnet3/systems/lid/task.py` | ESPnet2 `tasks/lid.py` の互換コピー。モデル構築の実装は同一。独自モデルを作り直さない。 |
| `espnet3/systems/lid/system.py` | `BaseSystem` 継承。LIDの統計収集のみ薄く切替。train/infer/measure/pack/uploadは共通処理を使う。 |
| `espnet3/systems/lid/collect_stats.py` | 生波形長が必要な場合に、ESPnet2 `collect_stats(model=None)` を呼ぶアダプター。共通統計処理へLID用分岐を増やさない。 |
| `espnet3/systems/lid/inference.py` | `Speech2Language`。Taskでモデルを読み、予測indexを言語コードへ変換。追加の `extract_embd:true` で正規化埋め込みを返す。既定の言語文字列APIは維持。 |
| `espnet3/systems/lid/metrics/accuracy.py` | 共通BaseMetricを利用。全体/macro指標、誤り一覧、言語別指標、誤り組合せ頻度。 |
| `espnet3/systems/lid/metrics/embedding.py` | SCP/NPYから言語別埋め込みと正規化平均を保存。t-SNEはESPnet2 `gen_tsne_plot` を再利用し、別アルゴリズムは実装しない。 |
| `egs3/TEMPLATE/lid/run.py` | ASR runnerをLIDにコピーする方針を採用。ASR runnerの引数変更を避け、LIDの7ステージを登録。設定読込・ステージ実行等は共通utility。 |
| `egs3/TEMPLATE/lid/conf/` | LIDの既定設定。利用可能なASR TEMPLATE設定を継承し、コーパス固有値は具体レシピへ置く。 |
| `egs3/voxlingua107/lid/dataset/` | VoxLingua固有Builder/Dataset、言語コード対応、manifest・lang2utt・category2uttの作成。 |
| `egs3/voxlingua107/lid/conf/` | MMS+ECAPA、catpow等の具体的な構成。TEMPLATEからの差分を記載。 |
| `egs3/voxlingua107/lid/src/inference.py` | hyp/ref/任意embeddingを共通writerの形式へ整える。 |
| `egs3/voxlingua107/lid/src/external_data.py` | 既存ESPnet2評価metadataのラベル整形。音声を複製・変換せず、新規出力先を使う。 |
| `egs3/voxlingua107/lid/src/external_dataset.py` | ESPnet2 SoundScpReaderで整形済みwav.scpを読み、モデルが持つ言語との交差集合を評価する。 |
| `test/espnet3/systems/lid/`, `test/egs3/voxlingua107/` | 統計、推論、評価、Builder、設定、外部metadataの契約を検査。 |
| `.agent/` | 開発方針とAPI・epoch契約の説明。引き継ぎ先でも読む。 |

登録ステージは `create_dataset → collect_stats → train → infer → measure → pack_model → upload_model`。
推論だけならcreate_dataset/collect_stats/trainを実行する必要はない。
今回の検証ではpack/uploadを実行しておらず、Hubへの公開もしていない。

### 統計とID

`model.model_conf.extract_feats_in_collect_stats:false` のとき、LID側は前処理を無効にして連結Datasetを走査し、
`([整数indexの文字列], {"speech": バッチ次元付きTensor})` をESPnet2の統計関数へ渡す。
生成する `speech_shape` の長さは**生波形のサンプル数**であり、mel bin数・特徴フレーム数ではない。
モデルを作らないため特徴量の平均/分散は生成しない。有効/省略時はBaseSystemの統計経路を使う。

2026-09-19更新: 同じ走査で `category2utt` と `lang2utt` を `${stats_dir}/{train,valid}/` に生成し、
複数Datasetでもshapeとカテゴリの連結後IDを一致させる。学習用preprocessor・sampler・推論inventoryはこの生成物を参照する。
`dataset2utt`/`utt2dataset` も出力し、Datasetの設定順を `0, 1, ...` のIDにする。
Datasetの構成・順序を変更したら統計を再収集する。言語が増える場合は `model.lang_num` も合わせる。

生成manifestはmini_an4 ASRと同様にレシピ側へ置く。既定は `${data_dir}/voxlingua107/{train,dev}/`。
元音声は `dataset_dir` で指定したNASから直接読み、コピー・変換しない。旧 `<source>/espnet3/` は削除せず、設定から参照しない。
音声パスの絶対パス仕様は維持する。別サーバーで音声を移設したら新しい `data_dir` にmanifestを再生成する。
`data_dir`/`stats_dir` を変更するときはtraining・inference・publicationの設定を合わせる。

推論はモデル設定・checkpointを直接読む。今回の公開モデルはutterance MVNなので、別途収集した全体平均/分散を必要としない。
これは「すべてのモデルで統計ファイル不要」という意味ではない。

Datasetのindexとmanifest/shape/categoryのIDを一致させる。サンプル辞書に未対応の `utt_id` を追加しない。
外部Datasetは元の発話IDを `utterance_ids` に保持し、出力SCPではソート後の整数indexを使う。

### 共通DataLoaderの差分と影響

現在のtracked差分はDataLoaderが+25/-4行、対応テストが+123行。既存テストは削除していない。

1. `catbel/catpow/catpow_balance_dataset` のときだけ既存 `build_category_batch_sampler` を呼ぶ。他方式は従来の `build_batch_sampler`。
2. カテゴリ方式にも `num_batches` による一覧の切り詰めを適用。
3. Lightningの0始まりepochを、ESPnet2 iter factoryへ渡す境界で `epoch + 1` にする。shard rotationは0始まりを維持。
4. category samplerの既定epochも同じ1始まり。明示的な `batches.epoch` は尊重。

理由は、ESPnet2のfactoryが1始まりepochで区間計算・shuffleを行うため。
**epoch変換はLID限定ではなく、ASR等のiter_factory経路にも適用される。**
旧0始まり実装と比べると、shuffle/worker seedや旧実験resume時のデータ順序が変わる。plain PyTorch DataLoaderは対象外。

`_target_` でiter factoryを選べても、現在のBuilderはその前にbatch一覧を構築するため、
factoryの設定変更だけではこのdispatchを置き換えられない。別の設定設計を採用するなら、初期batch生成と各epoch再構築の両方を検討すること。

**共通コードでのseed自動受渡しは追加していない。** LID TEMPLATEには整数の既定値0を明記し、
VoxLinguaレシピのtrain/validで `iter_factory.seed: ${seed}` を指定する。現在の既定値は3702。
以前のIterator seed=0とはデータ順序が変わる。過去の実験順序を維持する場合は両factoryに0を明示する。

## 3. ESPnet2 VoxLinguaとの共通点・相違

### 2026-09-19の修正検証

- LID・VoxLinguaレシピ・共通DataLoader: 134 tests passed。
- ASR・Base training・Trainer: 86 passed / 2 skipped。
- 2つの小規模WAV Datasetを実際のDataOrganizerで連結し、ESPnet2 collectorでshapeを生成。
  3種の実samplerで両Datasetが選択されること、IDと音声長、train/valid順序、worker数0/2を確認した。
- manifestを元音声と別ディレクトリへ作り、元音声を変更しないこと、公開bundleへのinventory同梱を確認した。
- 今回はCPUの回帰テスト。実VoxLingua全体やMMS/GPU学習を再実行した結果ではない。
- DDPのバッチ分配・loss重み付け・絶対音声パス・Builderの絶対importは変更していない。

### 既存の学習条件との差

モデル部品、LIDPreprocessor、samplerアルゴリズム、Adam/Tristage、raw speech長、t-SNE描画関数は既存ESPnet2を使う。
具体レシピはMMS-1B multilayer + ECAPA（512/1536）+ projector 192、AAMSoftmax、
catpow/batch_bins 2880000/最大16/upsampling 0.5、lr 5e-6、accumulation 2等を引き継いでいる。

ただし完全同一の実験条件と断定しない。

| 項目 | 現状の差・注意 |
| --- | --- |
| 学習ループ | ESPnet3はLightning。ESPnet2 Trainerそのものではない。 |
| Iterator | ESPnet2 CategoryIterFactoryに対し、現在のLID設定はSequenceIterFactory＋Builderのepochごとの再構築。 |
| seed | factoryにはrecipe設定から3702を渡す。学習全体の再開時乱数まで同一とは未検証。 |
| 精度 | recipeは `bf16-mixed`。ESPnet2の `use_amp:true` とビット単位で同じ処理を保証しない。 |
| DDP | 4 GPUテストでは `strategy: ddp_find_unused_parameters_true` を明示。recipe既定はまだauto。未使用パラメータがあるMMS構成でそのまま同条件になるとは限らない。 |
| checkpoint選択 | ESPnet2既定推論はbest、ESPnet3 recipeは2best平均。今回の公開モデル検証は明示的に公開bestを指定。 |
| t-SNE上限100件/言語 | ESPnet2は推論時の制限、ESPnet3は全件推論後の集約・描画の制限。 |
| 推論再開 | ESPnet2の途中checkpoint/resume/分散推論との完全一致は未検証。共通runnerの完了shard再利用は別機能。 |

## 4. 実施した検証

### 公開ESPnet2モデルによる今回の検証

以前取得した `espnet/lid_voxlingua107_mms_ecapa` の **valid.accuracy.best.pth** を使用。
MMSの初期重みだけでなく、学習済み107言語LIDモデル全体をロードした。再学習・モデル改変なし。
入力が3言語でも出力候補を3言語に絞らず、107クラスのまま評価した。

- GPU 0のみ、FP32、batch 1、16 kHz monoの全発話。GPU 5は不使用。
- 通常CLIの `infer measure`、両Datasetとも新しい外部Datasetアダプターを使用。
- `config.smoke.yaml` は元公開configに対し `lang2utt` パス1行だけ変更した既存コピー。
- 言語対応表は以前の検証で用意した107言語inventory。行順がclass indexなので勝手に並べ替えない。
- VoxLinguaは前回の独・英・日50件（公式dev由来）。
- FLEURSは既存 `egs2/fleurs_cs_lid/asr1/data/test_fleurs_lid` から、utt2lang順に独・英・日各10件。
  公式testの実音声だが、公開READMEの評価集合全体と同一選択かは未確認。

| 集合 | 件数 | 正解 | Accuracy | 埋め込み・t-SNE |
| --- | ---: | ---: | ---: | --- |
| VoxLingua部分集合 | 50 | 46 | 92.0% | 成功 |
| FLEURS部分集合 | 30 | 30 | 100.0% | 成功 |

80件すべてをESPnet2 `LIDTask.build_model_from_file` → model forwardでも別途実行し、
同じFP32・batch 1・全発話条件で、**予測が全件一致、正規化埋め込みの最大絶対差0**を確認した。
ESPnet2の推論CLI・AMP・分散・途中resume全体の一致を検証したわけではない。

確認した成果物: hyp/ref/embedding SCP、全80件の192次元有限・L2正規化埋め込み、
言語別NPZ/正規化平均NPZ、言語別・誤り頻度JSON、両集合の発話/平均t-SNE PNG/CSV。
Plotly/adjustText未導入のため対話型HTMLとラベル位置調整は未実施。
3言語平均の3点の図は機能確認用で、分離性能の根拠にはしない。
この成績は小規模テストであり、公開ベンチマークの再現値ではない。

公開checkpointのSHA-256:

```text
df3813d2fa05720e4dc21912c52a7402cf59f6f83ab2588f64d01c2074630a58
```

### 以前の4 GPU・3言語学習

MMS-1Bから新規3言語分類器で、train500/valid50/test50（3言語合計）を使用。
GPU 0–3、BF16、2epochで各rank 128 backward・64実optimizer更新、非有限勾配なし。
保存checkpointのAdam stepとMMS/ECAPA等の代表重みの変化も独立に確認した。

ただし、学習後に検証用callbackがCPU tensorへNCCL通信を呼んで失敗し、学習コマンドの終了コードは1。
学習本体の更新・checkpoint保存と、その後の検査エラーは分けて扱う。rank間の最終重み一致は未確認。
callback修正後のCPU回帰は合格したが、修正版による学習全体の再実行は未実施。
保存済みモデルのinfer/measureは正常終了し、test30/50正解だった。
これは上の公開107言語モデルの結果とは別実験。

### 回帰テスト

前回の移植検証ではLID/recipe/共通DataLoader関連130件、ASR system/inference/metricsと共通trainer関連56件が合格。
今回の公開モデル実行で製品コード変更はなかったため、全単体テストの再実行はしていない。

再確認例（リポジトリrootで、依存導入済みのPythonを使用）:

```bash
python -m pytest -q test/espnet3/systems/lid test/egs3/voxlingua107/lid \
  test/espnet3/components/data/test_dataloader_builder.py
python -m pytest -q test/espnet3/systems/asr/test_system.py \
  test/espnet3/systems/asr/test_asr_inference.py \
  test/espnet3/systems/asr/metrics/test_metrics.py \
  test/espnet3/components/trainers/test_trainer.py
```

ASRの一部テストはcwdに作業ファイルを作るため、read-only checkoutで実行する場合は書込可能なcwdと絶対テストパスを使う。

## 5. 別サーバーへ持っていくもの

### ソースとGit

`git diff` だけでは新規LIDソースを運べない。次のディレクトリ・ファイルを明示的に含める。
別worktreeの `.git` は元repoへの参照ファイルなので、それだけコピーして独立repoになると思わないこと。

```text
espnet3/components/data/dataloader.py
test/espnet3/components/data/test_dataloader_builder.py
espnet3/systems/lid/
egs3/TEMPLATE/lid/
egs3/voxlingua107/lid/
test/espnet3/systems/lid/
test/egs3/voxlingua107/
.agent/
```

**`egs3/voxlingua107/lid/dataset/` の4ソースはGit ignoreルールに一致する。**
`.gitignore` の `egs*/*/*/data*` に一致するため、LID追加コミットでは下記4ファイルを明示的に追加する。
引き継ぎ先で `git ls-files egs3/voxlingua107/lid/dataset` に4ファイルが含まれることを確認する。
データ・checkpoint・一時テスト成果物はコミット対象にしない。GitHubへのpushは別操作。

```text
dataset/__init__.py
dataset/builder.py
dataset/config.yaml
dataset/dataset.py
```

移動先では基準HEADだけでなく、利用中のESPnet2 LID実装を含む同じfork履歴を用意する。
公式mainへ一部ファイルだけコピーした構成は未検証。

### 重み・データ・成果物

| 対象 | 現サーバーの場所 |
| --- | --- |
| VoxLingua ZIP/展開音声 | `/mlnas/mitsumori/dataset/voxlingua107` |
| 公開モデルのconfig/checkpoint | 上記の `pretrained/lid_voxlingua107_mms_ecapa/exp_voxlingua107_raw/lid_mms_ecapa_baseline_raw/` |
| 107言語inventory | 上記の `smoke/lang2utt` |
| Hugging Face cache | 上記の `pretrained/huggingface/` |
| NLTK cache | 上記の `pretrained/nltk_data/` |
| 今回の公開モデル検証結果 | 上記の `smoke/public_lid_eval_20260917_eY3PVt/` |
| 前回の4 GPU学習・検証 | 上記の `smoke/mms_three_small_20260917_yxGNIv/` |
| FLEURS元metadata | `/home/mitsumori/espnet/egs2/fleurs_cs_lid/asr1/data/test_fleurs_lid/` |
| FLEURS音声 | 同recipeの `downloads/fleurs/all_materialized/audio/` |

公開モデル検証の保存先にはconfig、data/wav.scp・utt2lang、ログ、verification.json、検証スクリプト、
推論成果物、データ由来の情報を保存。音声と3.9GBの公開checkpointは重複コピーしていない。
前回学習の詳細はその実験の `verification/REPORT.md` を参照。

manifest/wav.scp/embedding.scpや保存configには絶対パスがある。
特に今回の実行時rootは `/tmp/espnet3-public-lid-eval.eY3PVt`。
NASへ保管しただけでは参照先は自動変更されない。移動先で音声パス・モデルパス・出力パスを明示的に更新し、
再実行は**新しいinference_dir**にする（古い完了shardを誤って再利用しない）。

MMSのHF cache revision: `0d2f7adb9903d98894d70ae11f7fbdfc8cb71a69`。
cache移動ではsnapshotsだけでなくblobs/refsとsymlink関係も保存する。
モデル・データ利用条件は元の配布条件に従う。

### 確認した環境

Docker image: `espnet_v2_snapshot_20260518`、Python 3.10.14、RTX A6000。
PyTorch 2.9.1+cu126、Lightning 2.6.1、S3PRL 0.4.18、Transformers 5.8.0、
Hydra 1.3.2、NumPy 2.2.6、SoundFile 0.13.1、scikit-learn 1.7.2、
pandas 2.3.3、matplotlib 3.10.9。
バージョンを記したのは実行環境の記録であり、この組合せ以外を非対応とする指定ではない。
全依存は元の環境をexportする等して別途保存し、espnetの通常セットアップも必要。

Dockerではcheckout、Python環境、モデルcache、音声、出力先をmountする。
入力はread-only、出力先のみ書込可でよい。ホストからNASを書けるかと、コンテナ内UIDで書けるかは別に確認する。
GPU番号は移動先で再確認。元サーバーではGPU 5を使わない指示を守った。

## 6. 公開モデル評価の再開手順

1. 上記ソース・公開モデル・107言語inventory・音声・cacheを移動。
2. 保存した `data/voxlingua50`, `data/fleurs30` のwav.scpを移動先の音声パスへ更新。
3. 保存 `inference.yaml` のrecipe_dir、exp_dir、modelの3つのパス、各data_dirを更新。
   `extract_embd:true`、FP32、batch 1、107言語のclass順を維持する。
4. `metrics.yaml` のrecipe_dir、exp_dir、inference_dirを同じ実験に合わせる。
5. repo rootがimportできる環境で、**学習設定を渡さず**実行する。学習設定を渡すとそのexp_dirが優先される。

```bash
export PYTHONPATH=/path/to/espnet3-dev
export HF_HOME=/path/to/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/path/to/nltk_data
python -m egs3.voxlingua107.lid.run --stages infer measure \
  --inference_config /path/to/evaluation/inference.yaml \
  --metrics_config /path/to/evaluation/metrics.yaml
```

オフライン指定はcacheを全て移動した場合に使う。MMS初期構築用の依存cacheも必要になる。
同じ80件のモデル呼び出し比較を再実行する場合は保存 `verify_public_model.py` の
ROOT/PUBLIC/INVENTORY定数も移動先へ変更する。
結果のverification.jsonでモデルパス、80件の一致、埋め込み差、バージョンを確認する。

## 7. 残っている制限・次の作業

- stockBuilderは展開済み107言語全体+devを要求し、自動downloadしない。
  前回の3言語学習は専用manifestアダプターであり、stockBuilderを3言語対応にしたものではない。
- DataOrganizerの複数Dataset連結は共通機能だが、現在のVox category2uttは単一splitの整数ID。
  複数Datasetのglobal offsetを反映したcategory対応表の自動生成は未実装。
- 今回の実外部評価はFLEURS30件のみ。Babel/ML-SUPERB/VoxPopuliの全量検証や、
  公開実験と同じ外部評価splitの再構築は完了していない。
- 外部Datasetは16 kHzの整形済み発話wavを期待する。8 kHz、pipe、segmentsはESPnet2 formatterで事前処理する。
  言語intersectionはclosed-set評価であり未知言語検出ではない。
- 4 GPUで確認したDDP strategyの本番反映は別途判断する。factory seedは上記のとおり設定で対応済み。
  multi-node、rankごとの可変batchに対する重み付け、端数batchの分配を含む一般条件を保証したテストではない。
- best/平均checkpointの選択と自動resumeは別途確認が必要。今回明示指定したcheckpointでの結果を、
  recipe既定の平均checkpointやresumeの検証済みという意味に広げない。
- 前回の検査callback修正後に、学習コマンドが終了コード0まで完走する再テストが残る。
- 長時間学習の収束、全107言語の性能、公開ベンチマーク再現、pack/uploadの実サービス試験は未完了。

本番の既存実験へ戻す前に、この文書の「検証済み」と「未検証」を分けて扱うこと。
