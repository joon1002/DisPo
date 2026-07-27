"""
논문용 훈련 쿼리 수 Ablation 선그래프
  - x축: 훈련 쿼리 수 (100, 200, 300, 500, 800)
  - y축: ASR (%)
  - 두 라인: ND-ASR (No Defense), RD-ASR (RAGDefender)
  - 출력: q_ablation_fullcorpus.pdf / .png (300 dpi)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 데이터 ──────────────────────────────────────────────
# randomq300은 훈련 실패(n_valid=0)이므로 randomq300_v2 사용
x_vals  = [100, 200, 300, 500, 800]
nd_vals = [66,  52,  49,  90,  10]   # ND-ASR (%)
rd_vals = [52,  33,  33,  87,  11]   # RD-ASR (%)

x_labels = ["100", "200", "300", "500", "800"]

# ── 스타일 설정 ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.6,
    "lines.markersize":  6,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ── 그림 크기 (ACL/EMNLP 단컬럼: ~3.3인치) ──────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6))

x_pos = np.arange(len(x_vals))

# ND-ASR
ax.plot(x_pos, nd_vals,
        color="#2166AC", marker="o", markersize=5,
        linewidth=1.6, label="ND-ASR", zorder=3,
        markerfacecolor="white", markeredgewidth=1.6)

# RD-ASR
ax.plot(x_pos, rd_vals,
        color="#D6604D", marker="s", markersize=5,
        linewidth=1.6, label="RD-ASR", zorder=3,
        markerfacecolor="white", markeredgewidth=1.6,
        linestyle="--")

# 값 레이블: ND는 항상 위, RD는 항상 아래 → 겹침 없음
# q500(90↔87), q800(10↔11) 근접 포인트도 자연스럽게 분리됨
for i, (nd, rd) in enumerate(zip(nd_vals, rd_vals)):
    ax.annotate(f"{nd}%", xy=(x_pos[i], nd),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color="#2166AC")
    ax.annotate(f"{rd}%", xy=(x_pos[i], rd),
                xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, color="#D6604D")

# 수평 기준선 (보조)
ax.axhline(y=50, color="#AAAAAA", linewidth=0.5, linestyle=":", zorder=1)

# x축
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Number of Training Queries", labelpad=4)

# y축
ax.set_ylim(-5, 105)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax.set_ylabel("ASR (%)", labelpad=4)

# 격자
ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)

# 범례
leg = ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                edgecolor="#CCCCCC", handlelength=1.8,
                handletextpad=0.5, borderpad=0.5,
                labelspacing=0.3)
leg.get_frame().set_linewidth(0.6)


plt.tight_layout(pad=0.3)

# ── 저장 ────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "q_ablation_fullcorpus.pdf")
png_path = os.path.join(out_dir, "q_ablation_fullcorpus.png")

fig.savefig(pdf_path)
fig.savefig(png_path)

print(f"[saved] {pdf_path}")
print(f"[saved] {png_path}")
