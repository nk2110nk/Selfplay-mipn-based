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

expertモデル:

```bash
source .venv/bin/activate
python train.py \
  -a Boulware Linear Conceder Atlas3 \
  -i Laptop \
  --model-type expert \
  --total-timesteps 1100000 \
  --device auto
```

generalモデル:

```bash
python train.py \
  -a Boulware Linear Conceder Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  --model-type general \
  --general-domain EnergySmall_A \
  --total-timesteps 1100000
```

`general`では観測shapeとaction headを`general_domain`から決めます。各対象ドメインはそのshape以内である必要があり、余白を0でpaddingします。`relative_time`は常に最後の要素です。対象ドメインに存在しないissue/valueはaction maskで選択できません。

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

`--auto-entropy`を指定すると、公開実装のintent/price別entropy係数を一般化した「action head別alpha」を自動調整します。目標entropyは各状態で有効なcategory数から計算し、`--target-entropy-ratio`で比率を変更できます。固定alphaが論文既定、auto entropyはSL warm startを持たない本適応版で早期collapseを抑えるための選択肢です。

## Opponent PoolとPFSP

poolは最初にBoulware、Linear、Conceder、Atlas3で初期化されます。その後、定期評価で優位になった学習方策をsnapshotとして追加します。公開実装のthree-source samplingをSLなしへ適応し、各slotを既定でcurrent self-play 20%、scripted 30%、historical snapshot 50%から選びます。利用できないsourceの確率は残りへ正規化します。

PFSPは公開実装の式`p(A) ∝ P[A dominates M]`に従います。本3者SAOP版ではdominance確率を次の複合推定値で近似します。

```text
empirical = (opponent_wins + 0.5 * draws + 0.5) / (matches + 1)
score_signal = sigmoid(opponent_negotiation_score - learner_negotiation_score)
difficulty = ((1 - agreement_rate) + min(length / 80, 1)
              + max(0, 3 - social_welfare) / 3) / 3
P_dominates = 0.5 * empirical + 0.3 * score_signal + 0.2 * difficulty
PFSP(i) ∝ exp(pfsp_alpha * P_dominates(i))
```

最後に`uniform_mix`の一様分布を混ぜます。未評価、NaN、全要素同値、総重み0の場合は一様分布へ戻します。

論文に基づくnegotiation scoreは独立関数として実装しています。

```text
Sc = (1 - min(agreement_rate, 1 - epsilon))^(-utility)
     + score_length_weight * negotiation_length
     + score_welfare_weight * social_welfare
```

既定値は`epsilon=0.01`、length weight=`-0.005`、welfare weight=`0.1`です。snapshot追加ではagent utilityを先に比較し、差が`dominance_tolerance=0.01`以内ならこのscoreで決めます。

pool上限時はscripted opponentを保持し、snapshotを次の順で削除します: negotiation scoreが低い、選択回数が少ない、追加stepが古い。判定は決定的です。

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
- 論文のSL source 30%はscripted sourceへ置き換え、current self-play 20%とhistorical RL snapshot 50%は維持します。
- 公開実装のpolicy lossはコード上`Q - alpha log pi`をgradient descentしていますが、本実装は論文の目的式どおり`alpha log pi - Q`を最小化します。
- 公開実装は連続priceとcategorical intentに別alphaを持つため、本実装では各issue headとaccept/reject headに別alphaを持たせます。

実装は論文と公開リポジトリ（確認commit `9d646bcb940f0e85f2ccdba6722e6739a25dd88c`）の設計を参照して独立に記述しており、公式実装のソースコードはコピーしていません。

参考:

- 論文: `/home/nakata/electronics-15-02039-v2.pdf`
- DOI: <https://doi.org/10.3390/electronics15102039>
- 公式実装: <https://github.com/1476900445/alpha-nego-framework-v1.0>
