# Baseline N=4, q500 Poison Artifacts

이 문서는 NQ q500으로 학습하고 NQ val100에 대해 inference한 기본 N=4 악성 문서 artifact를 정리한다. 여기서 N=4는 `doc0_seed + doc1 + doc2 + doc3`이며, `doc1~doc3`이 모델이 생성한 추가 악성 문서다.

## Canonical files

| Variant | Canonical CSV | Source run output | Rows | Poison columns | SHA256 |
|---|---|---|---:|---|---|
| v7-cont / Contriever | `data/generated/pd_eval100_v7_cont_n4g8.csv` | `/data/joonhyung/nq/results/grpo_whitebox_v7_1.5b_run1/pd_eval100_v7.csv` | 100 | `doc0_seed`, `doc1`, `doc2`, `doc3` | `e73157d4fa6b297214573b797a1b3efc69c10d7ed3a00dabca11cfabfdebebef` |
| v7-e5 / E5-base | `data/generated/pd_eval100_v7_e5_n4g8.csv` | `/data/joonhyung/nq/results/grpo_whitebox_v7_e5_run2/pd_eval100_v7_e5_g8.csv` | 100 | `doc0_seed`, `doc1`, `doc2`, `doc3` | `6e5e7e69dbe14671cdb065ac82e1efdc195a475f4ca2c019f7a796735ba626f5` |
| merged seed | `data/generated/pd_eval100_merged_seed.csv` | `/data/joonhyung/nq/results/eval_csv/pd_eval100_v7_merged_g8.csv` | 100 | `doc0_seed`, `doc1` ... `doc6` | `c8299a404672586e68a873836d431070f8d6a60a9bd406dda1f4e3fed5b9a587` |
| merged noseed | `data/generated/pd_eval100_merged_noseed.csv` | derived from merged seed by dropping `doc0_seed` | 100 | `doc1` ... `doc6` | `f3a36ad1d5d82adc918613738f03052208c19b700119dff49aa172006e5afd77` |

두 canonical CSV는 현재 git tracked 상태이며, 위 source run output과 byte-level 동일하다. `*_noseed.csv` 파일은 seed 문서를 제거한 평가용 파생본으로, N=4 원본 artifact는 아니다.

Merged seed 파일은 위 두 N=4 파일의 생성 문서를 합친 N=7 artifact다. 모든 row에서 `doc0_seed`는 cont/e5가 공유하는 동일 seed이며, `doc1~doc3`은 v7-cont 생성 문서, `doc4~doc6`은 v7-e5 생성 문서다. Merged noseed 파일은 같은 생성 문서 6개만 남긴 평가용 파생본이다.

## Train setup

| Parameter | v7-cont / Contriever | v7-e5 / E5-base |
|---|---|---|
| Train input | `/data/joonhyung/nq/results/nq_500_pd_7b.csv` (500 rows) | same |
| Evaluation input | `/data/joonhyung/nq/results/nq100_validate.csv` (100 rows) | same |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | same |
| Surrogate LLM | `lmsys/vicuna-7b-v1.3` | same |
| Defense/diversity encoder | `paraphrase-MiniLM-L6-v2` | same |
| White-box retriever | `facebook/contriever` | `intfloat/e5-base-v2` |
| Retriever scoring | raw dot product, normalized as `(dot - 0.40) / (1.50 - 0.40)` | cosine with `query: ` / `passage: ` prefixes, normalized as `(cos - 0.70) / (0.95 - 0.70)` |
| Epochs | 3 | 3 |
| Group size G | 8 by current baseline scripts/docs | 8, confirmed by `infer_g8.log` |
| Total injected docs N | 4 | 4 |
| Generated docs | 3 (`--num_adv_docs 3`) | 3 (`--num_adv_docs 3`) |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` | same |
| LR / optimizer regularization | `1e-5`, weight decay `0.01` | same |
| GRPO controls | `ADV_CLIP=2.0`, `GRAD_CLIP=0.5`, `LAMBDA_KENDALL=0.30` | same |
| Generation length | min 80, max 160 new tokens | same |
| Sampling | temperature `0.85`, top-p `0.92`, repetition penalty `1.1`, no-repeat ngram `4` | same |
| MMR diversity | `MMR_LAMBDA=0.60` | same |
| Generator dtype | `float16` | `float16` |

## Notes

- v7-e5의 G=8 inference는 `/data/joonhyung/nq/results/grpo_whitebox_v7_e5_run2/infer_g8.log`에서 완료 로그가 확인된다.
- v7-cont canonical 파일명과 현재 baseline script/docs는 G=8 기준으로 정리되어 있다. 다만 같은 run directory의 오래된 `auto_eval.log`에는 G=3 헤더가 남아 있어, CSV 자체만으로 G 값을 복원할 수는 없다.
- v7-cont canonical CSV는 2026-07-06에 number correction이 적용된 `/data/joonhyung/nq/results/grpo_whitebox_v7_1.5b_run1/pd_eval100_v7.csv`와 동일하다.
- v7-e5 canonical CSV는 `/data/joonhyung/nq/results/grpo_whitebox_v7_e5_run2/pd_eval100_v7_e5_g8.csv`와 동일하다.
