# α-Nego-based

`α-Nego-based` は、論文 **α-Nego: Self-Play Deep Reinforcement Learning for Negotiation Dialogues** の考え方を、MiPN-basedと同じ3者間・複数論点SAOP環境へ適用した実装です。

これは論文の完全再現ではありません。論文が対象とする自然言語対話の `dialogue act + price` を、MiPN-basedと同じ「各論点のvalue index + accept/reject」のMultiDiscrete行動へ置き換えています。また、教師ありデータを使用しないため、Behavior CloningによるWarm StartとSL方策へのKL正則化は含みません。

## 構成

- `train.py`: DSAC学習、PFSP、pool評価、snapshot追加、checkpoint再開
- `test_negotiator.py`: 3つのstyleによる評価と互換TSV出力
- `dsac.py`: twin distributional critic、target critic、quantile Huber loss、soft update
- `policy.py`: 論点単位のcategorical actor、action mask、style価値集約
- `environment.py`: 3者間SAOP、OnehotObserve2nT互換観測、snapshot opponent
- `opponent_pool/`: 永続pool、PFSP、pool全体の定期評価
- `data_calculator/summary_data.py`: TSVの簡易集計
- `tests/`: unit testと学習・再開・評価smoke test

MiPN-basedのドメインXML、効用関数、`MySAOMechanism`、ルールベース相手を参照します。ドメインデータは複製しません。既定では隣接する `/home/nakata/MiPN-based` を使用し、別の場所を使う場合は `MIPN_ROOT` を指定します。

## セットアップ

Python 3.9を推奨します。

```bash
cd /home/nakata/α-Nego-based
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU版PyTorchを明示する場合:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

このサーバーのRTX 5090では、`torch 2.8.0`のCUDA 12.8対応wheelを使用します。通常のPyPI版`torch==2.8.0`もCUDA 12.8依存を含みます。CUDA版をPyTorch indexから明示する場合は次のとおりです。

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`tensorboard`が環境に入っている場合は、学習指標を`MODEL_DIR/tensorboard/`にも自動出力します。CSVログだけを使う場合、追加インストールは不要です。

## 学習

`-a/--agents`で指定したエージェントだけが初期scripted poolに入ります。名前は大文字小文字を区別せず、空白区切りとカンマ区切りの両方を使えます。

```bash
python train.py -a boulware,conceder,Linear -i Laptop ...
```

対戦ペアは既定で同一相手の重複を許可した順序なし組み合わせです。3体なら`AA, AB, AC, BB, BC, CC`の6通りになります。異なる2体はOpponent 1/2のslotに偏らないよう、選択後に配置をランダム化します。重複を禁止する場合は`--no-allow-duplicate-opponents`を指定します。

学習中はTransformer-basedと同じくターミナルに進捗ゲージ、経過時間、残り時間、step速度、episode数、pool size、utility、entropyを表示します。非表示にする場合は`--no-progress`を使います。

expertモデル:

```bash
source .venv/bin/activate
python train.py \
  -a Boulware Conceder \
  -i Laptop \
  --model-type expert \
  --no-allow-duplicate-opponents \
  --total-timesteps 100000 \
  --device auto
```

generalモデル:

```bash
python train.py \
  -a Boulware Linear Conceder Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  --model-type general \
  --general-domain EnergySmall_A \
  --compatible-domains Coffee Camera Lunch SmartPhone Kitchen \
  --total-timesteps 300000
```

`general`では、学習ドメイン、`general_domain`、`compatible_domains`の和集合から、論点位置ごとの最大value数と最大観測長を計算します。未知ドメインはネットワーク形状の確保にだけ使われ、学習セッションには入りません。余白は0でpaddingし、`relative_time`は常に最後の要素です。対象ドメインに存在しないissue/valueはaction maskで選択できません。

`--compatible-domains`を省略したgeneralモデルは、既定で`Coffee Camera Lunch SmartPhone Kitchen`のすべてに対応します。標準構成のaction spaceは`[7, 5, 5, 5, 5, 5, 2]`、観測長は181です。従来の`[5, 5, 5, 5, 5, 5, 2]` checkpointはネットワーク形状が異なるため、CoffeeとSmartPhoneへの対応には新しい設定での再学習が必要です。

`--num-envs`は複数の独立セッションをround-robinで進めます。DSACの更新とpoolは共有されます。

学習の再開では、元と同じissues、agents、model type、general domainを指定します。`--total-timesteps`は再開後の累計終了stepです。

```bash
python train.py \
  --resume results/Laptop_Boulware-Linear/20260924-120000-TA/AlphaNego_Negotiator \
  -a Boulware Linear \
  -i Laptop \
  --model-type expert \
  --total-timesteps 1200000
```

## 評価

同じcheckpointを3つのstyleで評価できます。

```bash
python test_negotiator.py \
  --model-path results/Laptop_Boulware-Linear/20260924-120000-TA/AlphaNego_Negotiator \
  --agents Boulware Linear \
  --issues Laptop \
  --episodes 100 \
  --style neutral

python test_negotiator.py -m MODEL_DIR -a Boulware Linear -i Laptop -e 100 --style aggressive
python test_negotiator.py -m MODEL_DIR -a Boulware Linear -i Laptop -e 100 --style conservative
```

styleの価値集約は次のとおりです。

- `neutral`: 全quantileの平均
- `aggressive`: `--aggressive-quantile`以上の上側平均 + `--risk-weight` × 標準偏差
- `conservative`: `--conservative-quantile`以下の下側CVaR

評価時は有効joint action数が`--candidates`以下なら全列挙し、より大きい場合はactorから候補を生成してcriticのstyle価値で順位付けします。Laptopは既定値64に対して54通りなので全列挙されます。

既存のcase構造へ同時出力する場合:

```bash
python test_negotiator.py -m MODEL_DIR -a Boulware Linear -i Laptop \
  -e 100 --style neutral --case 1 --export-root results_alpha-nego-based
```

## 一括実験

`run_command/`には、標準7ドメインとBoulware、Conceder、Linear、Atlas3を使う一括実験スクリプトがあります。expert学習は重複を含む順序なし10ペアを各ドメインで個別に学習するため、合計70モデルです。general学習は全ドメイン・全エージェントを1モデルへ登録します。

```bash
# expert: 70モデル、各100,000 step
CUDA_VISIBLE_DEVICES=0 ./run_command/train_expert.sh

# general: 1モデル、300,000 step
CUDA_VISIBLE_DEVICES=0 ./run_command/train_general.sh

# 各expertモデルを対応するドメイン・固定ペアで評価
CUDA_VISIBLE_DEVICES=0 ./run_command/test_expert.sh

# generalモデルを既知7＋未知5ドメイン x 10ペアで評価
CUDA_VISIBLE_DEVICES=0 ./run_command/test_general.sh
```

generalの未知ドメインは`Coffee`、`Camera`、`Lunch`、`SmartPhone`、`Kitchen`です。これらは学習対象に含めず、general checkpointの互換形状だけに反映します。Coffeeの第1論点は7値、SmartPhoneの第1論点は6値あるため、generalモデルの第1action headは7カテゴリで構築されます。

既定値は`DEVICE=cuda`、`SEED=0`、評価は`EPISODES=100`、`STYLE=neutral`です。スクリプトを編集せず、環境変数で変更できます。`DRY_RUN=1`はコマンド表示のみ、`LIMIT=N`はexpertの先頭Nケースだけを実行します。

```bash
DRY_RUN=1 LIMIT=3 ./run_command/train_expert.sh
DEVICE=cpu EPISODES=10 LIMIT=1 ./run_command/test_expert.sh
STYLE=conservative MODEL_PATH=/path/to/AlphaNego_Negotiator \
  ./run_command/test_general.sh
```

評価完了後、`Results_MultiNego`の共通構造へcase単位で配置できます。既定の`case1`では、expert 70件とgeneral 120件がすべて101行（header＋100 episodes）であることを確認してからコピーします。

```bash
./run_command/export_case_results.sh
CASE_NAME=case2 ./run_command/export_case_results.sh
```

## case1-case3本実験

`case1`から`case3`は保存先だけでなく、学習・pool評価・最終評価で使う効用順序を表します。

| case | 学習エージェント | Opponent 1 | Opponent 2 |
|---|---|---|---|
| case1 | utility1 | utility2 | utility3 |
| case2 | utility2 | utility3 | utility1 |
| case3 | utility3 | utility1 | utility2 |

本実験はexpert 210モデル（7既知ドメイン x 10相手ペア x 3 cases、各100,000 step）とgeneral 3モデル（7既知ドメイン x 3 cases、各300,000 step）です。seedは0、generalだけが5未知ドメインも評価します。結果は`results/case1`から`results/case3`へ分離されます。

2 GPUを使ってDocker内の学習をtmuxで開始します。

```bash
bash run_command/launch_case_experiments_tmux.sh
tmux attach -t selfplay-mipn-cases
```

各GPUのジョブはcheckpointが完了済みならskipし、未完了ならresumeします。手動で1 shardだけ動かす場合:

```bash
CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 SHARD_COUNT=2 \
  bash run_command/run_case_experiments.sh
```

全学習完了後の評価も2 shardへ分割できます。各条件100 episode、neutral styleで、expertは対応する既知ドメイン・相手ペア、generalは既知7・未知5ドメインの全10ペアを評価します。

```bash
CUDA_VISIBLE_DEVICES=0 SHARD_INDEX=0 SHARD_COUNT=2 \
  bash run_command/evaluate_case_experiments.sh
CUDA_VISIBLE_DEVICES=1 SHARD_INDEX=1 SHARD_COUNT=2 \
  bash run_command/evaluate_case_experiments.sh
```

共通集計用TSVは`results/caseN/evaluation/<expert|general>/<pair>/<domain>/caseN/`へ保存されます。

## DSAC

actorは公開実装のshared encoderをMultiDiscrete向けに適応し、各issueとaccept/rejectに独立したcategorical headを持ちます。actor/criticはLayerNorm、ReLU、dropoutを使用します。criticは状態とjoint actionのone-hotを入力し、既定で64 quantileを返します。

公開実装のAlgorithm 2に合わせ、distributional Bellman targetにはentropy項を入れず、target twin criticの各quantileについて小さい値を採用します。actor側ではcriticごとにstyle価値を計算して小さい方を使い、SACの`alpha * log pi - Q_style`を最小化します。離散行動の勾配にはGumbel-Softmax straight-through、entropyにはcategorical分布から求めた厳密値を使います。criticは毎step、actorは既定で2 critic updateごとに更新し、勾配clipを適用します。

既定値:

- actor learning rate: `3e-5`
- critic learning rate: `1e-4`
- batch size: `128`
- replay buffer: `1,000,000`
- quantiles: `64`
- target update τ: `0.005`
- entropy coefficient: `0.01`

entropy coefficientは固定値として`--entropy-coefficient`で設定します。checkpointにはactor、critics、target critics、optimizers、replay buffer、乱数状態を保存します。

`--auto-entropy`を指定すると、公開実装のintent/price別entropy係数を一般化した「action head別alpha」を自動調整します。目標entropyは各状態で有効なcategory数から計算し、`--target-entropy-ratio`で比率を変更できます。`--target-entropy-final-ratio`と`--entropy-anneal-steps`を指定すると、探索量を学習進行に合わせて線形に減少できます。固定alphaが論文既定、auto entropyはSL warm startを持たない本適応版で早期collapseを抑えるための選択肢です。

`--canonical-accept-action`でaccept時の無意味なbid成分をcritic入力から除きます。`--hierarchical-entropy`では、accept/reject headのentropyは常に保ち、bid headのentropyはreject確率で重み付けします。これにより、acceptする行動に存在しない「bidの多様性」を報酬しません。

## Opponent PoolとPFSP

poolは`-a/--agents`で指定したScripted opponentで初期化されます。その後、定期評価で優位になった学習方策をsnapshotとして追加します。既定では初期学習を指定したScripted opponentのみで行い、snapshot追加後はScripted/Historicalの基礎重みを50%ずつにし、PFSP重みと組み合わせてペアを抽選します。current self-playは既定で0%で、`--self-play-probability`を指定した場合のみ有効です。利用できないsourceの確率は残りへ正規化します。

PFSPは公開実装の式`p(A) ∝ P[A dominates M]`を「学習エージェントから見た難しさ」として近似します。相手の効用関数は非公開とし、合意の有無、自分のutility、交渉長だけを使います。

定期pool評価の`--pool-eval-episodes`（既定4）は最低評価回数です。expertでは従来どおり1ドメインを4回評価します。generalでは全学習ドメインを同数評価できる完全な周回へ切り上げるため、7ドメイン・既定4なら各ドメイン1回の合計7回、14なら各2回、28なら各4回です。集計は各ドメイン同数のmacro averageとなり、評価TSV末尾の`domain`列でcoverageを確認できます。

```text
difficulty = 0.45 * (1 - agreement_rate)
           + 0.40 * (1 - clip(agent_utility, 0, 1))
           + 0.15 * clip(negotiation_length / 80, 0, 1)
confidence = matches / (matches + 10)
P_dominates = confidence * difficulty + (1 - confidence) * 0.5
PFSP(i) ∝ exp(pfsp_alpha * P_dominates(i))
```

最後に`uniform_mix`の一様分布を混ぜます。未評価、NaN、全要素同値、総重み0の場合は一様分布へ戻します。

論文に基づくnegotiation scoreは独立関数として実装しています。

```text
Sc = (1 - min(agreement_rate, 1 - epsilon))^(-utility)
     + score_length_weight * negotiation_length
```

既定値は`epsilon=0.01`、length weight=`-0.005`です。snapshot追加ではagent utilityを先に比較し、差が`dominance_tolerance=0.01`以内ならこのscoreで決めます。

学習報酬は変更前と同じく合意時の自分のutilityだけです。相手のutilityとsocial welfareはactor観測、critic入力、学習報酬、PFSP、snapshot優位判定のいずれにも使用しません。ベンチマークの学習ログと評価TSVには事後評価用として記録しますが、意思決定には戻しません。

pool上限時はscripted opponentを保持し、snapshotを次の順で削除します: PFSPと同じ定義で難易度が低い（学習エージェントにとって簡単）、選択回数が少ない、追加stepが古い。無効な難易度は最優先で削除し、最後にIDでtie-breakするため判定は決定的です。追加判定時のbenchmarkは新snapshot自身との対戦結果ではないため、そのsnapshotの難易度には流用しません。新snapshotは追加直後のpruningから一度保護し、実際の対戦またはpool評価から難易度を蓄積します。

## 出力

既定の学習先:

```text
results/<domains>_<agents>/<YYYYMMDD-HHMMSS>-TA/AlphaNego_Negotiator/
```

主な出力:

```text
checkpoint.pt
config.json
training_log.csv
pool/pool.json
pool/snapshots/snapshot-<step>.pt
evaluation/step-<step>.tsv
csv/<agent0>-<agent1>/<domain>/det=False_noise=False/*.tsv
```

評価TSVの先頭7列は既存実装と同じです。

```text
my_util  opp_util1  opp_util2  social  nash  agreement  step
```

非合意時は3者のutilityを0にします。8列目以降にstyle、pool ID、source、negotiation score、seedを追加します。このためMiPN-basedの`data_calculator/summary_data.py`も先頭列をそのまま読めます。

簡易集計:

```bash
python data_calculator/summary_data.py --data-dir MODEL_DIR/csv --output summary.csv
```

## テスト

```bash
source .venv/bin/activate
python -m pytest -q
```

テストには7ドメインの読込、観測と行動、mask、replay buffer、quantile loss、style集約、PFSP、pool永続化とpruning、checkpoint再開、snapshot opponent、3style評価、互換TSVが含まれます。

## 論文との差分

- SLデータ、Behavior Cloning、SL Warm Start、KL正則化を使用しません。
- 自然言語生成・parser・dialogue actを使用しません。
- actionはMiPN互換のissue value + accept/rejectです。
- 初期poolはSL agentではなく4種類のscripted negotiatorです。
- 2者交渉ではなく、MiPN互換の3者間SAOPです。
- style選択は巨大なjoint actionを全列挙せず、actorが生成した候補をquantile criticで順位付けします。
- 論文のSL sourceは`-a`で指定したScripted opponentへ置き換えます。3体指定時の6組を初期学習に使うため、current self-playは既定で無効です。
- 公開実装のpolicy lossはコード上`Q - alpha log pi`をgradient descentしていますが、本実装は論文の目的式どおり`alpha log pi - Q`を最小化します。
- 公開実装は連続priceとcategorical intentに別alphaを持つため、本実装では各issue headとaccept/reject headに別alphaを持たせます。

実装は論文と公開リポジトリ（確認commit `9d646bcb940f0e85f2ccdba6722e6739a25dd88c`）の設計を参照して独立に記述しており、公式実装のソースコードはコピーしていません。

参考:

- 論文: `/home/nakata/electronics-15-02039-v2.pdf`
- DOI: <https://doi.org/10.3390/electronics15102039>
- 公式実装: <https://github.com/1476900445/alpha-nego-framework-v1.0>
