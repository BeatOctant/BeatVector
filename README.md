# OctantVector

> **To use BeatOctant** (install, known issues): [Getting Started](https://github.com/BeatOctant/OctantSpecs/blob/main/GettingStarted_en.md) · [はじめかた（日本語）](https://github.com/BeatOctant/OctantSpecs/blob/main/GettingStarted_ja.md)
>
> BeatOctant is written mostly with AI coding agents (Anthropic's Claude). The maintainer decides what to build and reviews every change. It does not generate maps; it recommends existing maps from BeatSaver. ([作り方（AI の使い方）](https://github.com/BeatOctant/OctantSpecs/blob/main/README.md#作り方ai-の使い方))
>
> OctantVector is the vector search library behind BeatOctant, a Beat Saber map recommender (the OctantPath mod for Beat Saber PC and OctantPath for Quest).
>
> BeatOctant is an unofficial fan project for Beat Saber. It is not affiliated with, endorsed by, or sponsored by Beat Games or Meta. Beat Saber and Meta Quest are trademarks of their respective owners. Beat Games・Meta とは関係のない非公式の MOD・ツールです。
>
> 報告と案は Discord の BeatOctant サーバー（バグは `#bug-reports`、案は `#feedback-ideas`）で受けます。外からの pull request は受けていません。

ローカルに保存した BeatSaver のマップアーカイブを、複数のエンコーダ（音声モデル・譜面パターン・メタデータなど）でベクトル化し、似たマップを探すためのツールです。プレイヤー自身のプレイデータを使った推薦の評価もできます。

計算のライブラリです。配布するベクトル DB を書き出す計算は OctantHarvest（配布 DB を作るホスト）から、推薦の計算は OctantCompass（ゲーム外の推薦サーバー）から呼ばれます。下のコマンドは、実験と確認のための開発用です。

仕様の文書はすべて [OctantSpecs](https://github.com/BeatOctant/OctantSpecs) にあります。このリポジトリにはセットアップと使い方だけを書きます。

- 憲法（責務の境界）、契約、全体の計画: [OctantSpecs](https://github.com/BeatOctant/OctantSpecs#文書の地図)
- 設計方針と判断の記録: [octantvector/design.md](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/design.md)
- エンコーダ一覧と既知の問題: [octantvector/encoders.md](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/encoders.md)
- スイングシミュレーション（難易度・プレイスタイル）: [octantvector/swing.md](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/swing.md)
- 設定（パス・環境変数）: [octantvector/config.md](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)
- 実験ログ: [octantvector/experiments.md](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/experiments.md)

## 組み合わせる版

支えるのは、各部品の最新の版どうしの組み合わせだけです（[decisions.md の全体と段階](https://github.com/BeatOctant/OctantSpecs/blob/main/decisions.md#全体と段階)、2026-10-22）。この README を書いた時点の相手は次のとおりです。

| 相手 | 版 | つなぐもの |
| --- | --- | --- |
| OctantHarvest（DB を書く計算を呼ぶ） | 最新 | [契約1 生成側（OctantHarvest → OctantVector）](https://github.com/BeatOctant/OctantSpecs/blob/main/contracts/engine-generation.md) 1.1、[契約4（OctantHarvest の収集結果 → OctantVector）](https://github.com/BeatOctant/OctantSpecs/blob/main/contracts/generation-input.md) |
| OctantCompass（推薦の計算を呼ぶ） | 最新 | [契約1 推薦側（OctantCompass → OctantVector）](https://github.com/BeatOctant/OctantSpecs/blob/main/contracts/engine-runtime.md) 1.0、[契約2（DB 生成 ↔ 推薦コア）](https://github.com/BeatOctant/OctantSpecs/blob/main/contracts/vector-db.md) 1.1 |
| OctantVectorPlugins | `PLUGIN_API` 0.3 のもの（いまの main） | `octantvector.plugin` |

このリポジトリの今の版は v0.10.0 です。

## セットアップ

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -e ".[embed,dev]"
```

音や画像のモデルを使うビュー（CLAP、MERT、WavLM とそのボーカル版、Whisper、SigLIP）は [OctantVectorPlugins](https://github.com/BeatOctant/OctantVectorPlugins) のプラグインにあり、使うものだけ別に入れます（`[vocal]` はボーカル版も計算するときだけ）。プラグインは `octantvector.plugin` の版（`PLUGIN_API`。いまは 0.3）が同じものだけが読み込まれ、違えば `check` とビューを使うときのエラーに理由が出ます:

```bash
.venv/bin/pip install "octantvector-clap[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=clap"
```

```bash
.venv/bin/pip install "octantvector-mert[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=mert"
```

```bash
.venv/bin/pip install "octantvector-wavlm[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=wavlm"
```

ボーカルのビュー（`stem = "vocal"`）も計算するなら `".[embed,vocal]"` にします（Demucs で歌声を分離します）。モデルのファイルは OctantVector が取りに行かず、`models_dir`（環境変数 `BV_MODELS_DIR`）の下の `hf/<repo>/<revision>/` から読みます（`octantharvest models` が置きます）。検索と評価だけなら `pip install -e .` でも動きます（torch があれば GPU を使います）。アーカイブとカタログの場所を [設定の文書](https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)に従って設定してから、状態を確認します。

```bash
.venv/bin/octantvector check
```

OctantCompass と OctantHarvest は、このリポジトリの版のタグ（`v0.10.0` の形）から `pip install "octantvector @ git+https://github.com/BeatOctant/OctantVector@v0.10.0"` で入れます（[契約1（OctantCompass/OctantHarvest to OctantVector）の版の付け方](https://github.com/BeatOctant/OctantSpecs/blob/main/contracts/engine-runtime.md#版の付け方)）。タグを付ける前の変更を使うときは、そのコミットから入れます（`@<コミット>`。タグはいくつかの変更をまとめて付ける。2026-10-19）。

## 使い方

更新（カタログとアーカイブを台帳に取り込み、名前を指定したエンコーダで足りない分だけ計算します。再実行すると未計算の分だけ続きから計算します。毎日の実行は OctantHarvest がします）:

```bash
.venv/bin/octantvector update clap_htsat_unfused meta_tags chart_patterns npsnjs_std
```

計算する範囲はプールで選びます（`--pool`、既定は全量の `distribution`）。ビューを試すときは、全体から 20,000 件を決まった種で選ぶ `trial` を使います。プールは設定の `[pools.<名前>]` 表で変えたり足したりできます（キーは `min_upvotes`・`standard`・`sample`・`seed`・`include`。`include` は必ず入れるマップの ID かハッシュを 1 行に 1 つ書いたファイルです。`src/octantvector/generation/pools.py`）:

```bash
.venv/bin/octantvector update clap_htsat_unfused --pool trial
```

OctantHarvest からは契約1の `compute(input, store, views, pool="distribution", models_dir=...)` で同じ計算をします（`views` は必ず渡し、`models(views)` が読むモデルの一覧を返します）。

コードを変えた前後でベクトルが同じかは、古いストアと新しいストアを比べて確かめます（ビューごとのコサイン類似度と最大の差、悪い 10 件）:

```bash
PYTHONPATH=src .venv/bin/python -I spikes/compare_views.py old/store store clap_htsat_unfused@1 mert_v1_95m@1
```

検索（BeatSaver のキーで指定。`--views` でエンコーダと重みを必ず指定します。プラグインのビューはそのプラグインを入れたときだけ使えます）:

```bash
.venv/bin/octantvector search 1fef --views clap_htsat_unfused:0.5,meta_tags:0.5 --k 10
```

正規化の統計を固定する（以後の検索と評価に `--calibration` で渡します。マップが増えても座標が動きません）:

```bash
.venv/bin/octantvector calibrate clap_htsat_unfused meta_tags --pool trial
```

ビューが欠けたマップも残して検索し、ビューごとの上位も出す:

```bash
.venv/bin/octantvector search 1fef --views clap_htsat_unfused:0.5,meta_tags:0.5 --missing reweight --per-view
```

配布物の次の版を試しに書き出す（書き出し先は `<store>/releases/`）。ビューは毎回指定します。最初の版は校正も指定し、2 回目からは前の版との差分も作ります。毎日の実行、承認、公開は [OctantHarvest](https://github.com/BeatOctant/OctantHarvest) が受け持ちます:

```bash
.venv/bin/octantvector release clap_htsat_unfused meta_tags --calibration store/calibration/<校正 ID>.json
```

```bash
.venv/bin/octantvector release clap_htsat_unfused meta_tags
```

`--notice <Markdown のファイル>` を付けると、DB 全体の利用条件とクレジットとしてマニフェストの `notice` に入ります（付けなければ前の版のものを引き継ぎます）。ビューごとのクレジット（モデルや移植したコードの出所）は、エンコーダの `credits` からマニフェストのビューの `credits` に自動で入ります。

ビューの名前に `beatsaver_mappers` を入れると、カタログの譜面の作者（BeatSaver のアカウント ID。名前は使わない）から関係のビュー `beatsaver_mappers@1`（役割は `mapper`）を作って DB に入れます。計算と校正はいらず、入れるのは `release` だけです。

設定の `artists`（`BV_ARTISTS`）に OctantHarvest が書き出したアーティストの 2 つのファイル（`map_artists.jsonl.gz`、`similar_artists.jsonl.gz`）のフォルダを指定すると、近いアーティストの関係のビュー `listenbrainz_artists@<YYYYMMDD>`（役割は `artist`。日付は ListenBrainz から取った日）が DB に入ります。指定しなければ前の版のものを引き継ぎます。

OctantCompass と同じ手順で、差分を当てて開き、検索する（譜面のビューは `1fef:OneSaber:Expert` のように譜面を指定でき、マップだけなら最難の Standard 譜面で検索します。譜面のビューは同じプレイスタイルの譜面だけを返します）:

```bash
.venv/bin/octantvector apply store/releases/2026.10.02 store/releases/2026.10.03.diff host/2026.10.03
```

```bash
.venv/bin/octantvector db host/2026.10.03 1fef --k 10
```

カタログだけを使った近傍品質の評価:

```bash
.venv/bin/octantvector eval-neighbours --views clap_htsat_unfused:1
```

結果には、起点の upvote ごとの内訳（`by_upvotes`）も入ります。`--isolation-view` を付けると、起点がそのビューでどれだけ孤立しているか（近い k マップとの平均の近さ）ごとの内訳（`by_isolation`）も出ます。ビューの組み合わせを比べるときは `--isolation-view` を同じにします（同じ起点の組で比べるため）:

```bash
.venv/bin/octantvector eval-neighbours --views clap_htsat_unfused:1,mert_v1_95m:1 --isolation-view clap_htsat_unfused:1
```

自分のプレイデータでの評価（データはローカルで読むだけで、出力先は自分で指定します）:

```bash
.venv/bin/octantvector playdata --votes votedSongs.json --player-data PlayerData.dat --datarecorder beatsaber.db --out my_signals.json
```

```bash
.venv/bin/octantvector eval-personal --signals my_signals.json --views clap_htsat_unfused:1,meta_tags:1 --beta 1
```

スイングシミュレーション（1 マップの各スイングと評価値、教師データの作成と評価、スタイルタグのモデル作成）:

```bash
.venv/bin/octantvector swing-trace 1fef --swings
```

```bash
.venv/bin/octantvector swing-dataset --out swing_rows.jsonl.gz
```

```bash
.venv/bin/octantvector swing-eval --rows swing_rows.jsonl.gz
```

```bash
.venv/bin/octantvector swing-train --rows swing_rows.jsonl.gz --out swing_model.json
```

## テスト

```bash
.venv/bin/python -m pytest
```

テストは合成データだけを使うので、GPU・モデル・アーカイブなしで実行できます。

pull request ごとに GitHub Actions が同じテストを動かします（`.github/workflows/test.yml`）。文書だけの変更では動かず、同じ pull request に続けて push すると前の実行を止めます。

## ライセンス

[MIT](LICENSE)（2026-10-12）。ほかから移したコードの元のライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にあります。
